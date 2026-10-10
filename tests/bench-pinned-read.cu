// bench-pinned-read：GPU 读锁页主机内存的带宽，按（分配方式 × 分配线程所在 NUMA 节点 × 填充线程所在 NUMA 节点）列表。
//
// 目的：核实 Strata-AC922（refs/strata-ac922/docs/IBM_AC922.md "Hardware rules learned on the AC922"）的规则——
//   * GPU 读的锁页内存必须来自 cudaHostAlloc，且由 GPU 所在 NUMA 节点上的线程分配（它忽略 numactl --membind）：
//     本 socket 68～72 GB/s，跨 socket 40；
//   * 页面要由它所在节点的 CPU 填充：被另一 socket 最后写过的本地页只有 45 GB/s；
//   * cudaHostRegister 的内存在 UVM 迁移开启时 0～48 GB/s 不稳。
// 对照我们 moe-meas-0.md 测到的 34～35 GB/s（numactl --membind=<节点> + 主线程 cudaHostAlloc + 主线程 memset）。
//
// 编译（ac922-3，CUDA 12.4）：
//   /usr/local/cuda-12.4/bin/nvcc -O3 -arch=sm_70 -std=c++17 -o bench-pinned-read bench-pinned-read.cu -lpthread
// 用法：
//   bench-pinned-read [--mib 1024] [--reps 5] [--gpus 0,3] [--modes hostalloc,mallochost,register,policy,nobind]
//                     [--no-matrix] [--concurrent] [--small]
//   分配方式：
//     hostalloc  线程绑到“分配节点”的 CPU 后 cudaHostAlloc(Mapped|Portable)；
//     mallochost 同上，cudaMallocHost；
//     register   mmap 匿名页，由“填充节点”线程首次触碰（页落在该节点），再由“分配节点”线程 cudaHostRegister(Mapped|Portable)；
//     policy     我们 ggml-cuda.cu 现在的做法：分配线程绑到“分配节点”以外的节点、set_mempolicy(MPOL_BIND 分配节点)、
//                cudaHostAlloc、然后 mbind(MPOL_MF_MOVE) 搬页；打印搬页前后的页面分布；
//     nobind     moe-meas-0 的测法：主线程直接 cudaHostAlloc + 主线程 memset（节点由外部 numactl 决定）。
//   每一行：GPU、GPU 所在节点、方式、分配节点、填充节点、move_pages 抽样到的页面节点分布、直读 GB/s、cudaMemcpyAsync H2D GB/s。
//   --concurrent：所有 GPU 同时读，三种放置（每卡本地节点分配+填充 / 全部节点 0 / 全部节点 8），打印每卡与合计。
//   --small：小块（20 B～128 MiB）在另一节点线程 + set_mempolicy(MPOL_BIND 目标节点) 下连续分配，打印地址的页内偏移、
//            move_pages 节点分布与 mbind(MPOL_MF_MOVE) 的返回值（策略是否足以放对节点；小块是否从共享 2 MiB 映射切出、mbind 是否 EINVAL）。
//
// 只用 Linux 系统调用（set_mempolicy/mbind/move_pages），不依赖 libnuma。

#include <cuda_runtime.h>

#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

#ifndef MPOL_DEFAULT
#define MPOL_DEFAULT 0
#endif
#ifndef MPOL_BIND
#define MPOL_BIND 2
#endif
#ifndef MPOL_MF_MOVE
#define MPOL_MF_MOVE (1 << 1)
#endif

#define CK(x) do {                                                                     \
    const cudaError_t err_ = (x);                                                      \
    if (err_ != cudaSuccess) {                                                         \
        fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__, __LINE__,          \
                cudaGetErrorString(err_), #x);                                         \
        exit(1);                                                                       \
    }                                                                                  \
} while (0)

// ---------------------------------------------------------------- sysfs / NUMA 辅助

static std::vector<int> parse_list(const char * buf) {
    std::vector<int> ret;
    char tmp[512];
    snprintf(tmp, sizeof(tmp), "%s", buf);
    for (char * tok = strtok(tmp, ",\n"); tok != nullptr; tok = strtok(nullptr, ",\n")) {
        int first = -1, last = -1;
        if (sscanf(tok, "%d-%d", &first, &last) == 2) {
            for (int n = first; n <= last; n++) ret.push_back(n);
        } else if (sscanf(tok, "%d", &first) == 1) {
            ret.push_back(first);
        }
    }
    return ret;
}

static std::string read_file(const char * path) {
    FILE * f = fopen(path, "r");
    if (!f) return "";
    char buf[1024] = {};
    const size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = 0;
    return buf;
}

// 有 CPU 的在线节点（排除 POWER9 上显存对应的 250～255）
static std::vector<int> host_nodes() {
    std::vector<int> ret;
    for (int node : parse_list(read_file("/sys/devices/system/node/online").c_str())) {
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node);
        const std::string cpus = read_file(path);
        if (!parse_list(cpus.c_str()).empty()) ret.push_back(node);
    }
    return ret;
}

static bool node_cpuset(int node, cpu_set_t & set) {
    CPU_ZERO(&set);
    if (node < 0) return false;
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node);
    for (int c : parse_list(read_file(path).c_str())) {
        if (c < CPU_SETSIZE) CPU_SET(c, &set);
    }
    return CPU_COUNT(&set) > 0;
}

static int max_node() {
    int m = -1;
    for (int n : parse_list(read_file("/sys/devices/system/node/possible").c_str())) m = std::max(m, n);
    return m >= 0 ? m : 1023;
}

// 节点 node 的单节点掩码；返回内核期望的 maxnode
static unsigned long nodemask_one(int node, std::vector<unsigned long> & mask) {
    const int bits = 8 * (int) sizeof(unsigned long);
    mask.assign(max_node()/bits + 2, 0);
    mask[node/bits] |= 1ul << (node % bits);
    return (unsigned long) mask.size() * bits + 1;
}

static int gpu_node(int dev) {
    char bdf[32] = {};
    if (cudaDeviceGetPCIBusId(bdf, sizeof(bdf), dev) != cudaSuccess) { (void) cudaGetLastError(); return -1; }
    for (char * p = bdf; *p; p++) if (*p >= 'A' && *p <= 'Z') *p = (char) (*p - 'A' + 'a');
    char path[160];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/numa_node", bdf);
    const std::string s = read_file(path);
    return s.empty() ? -1 : atoi(s.c_str());
}

// 当前线程所在 CPU 的节点
static int node_of_cpu(int cpu, const std::vector<int> & nodes) {
    for (int n : nodes) {
        cpu_set_t s;
        if (node_cpuset(n, s) && CPU_ISSET(cpu, &s)) return n;
    }
    return -1;
}

// 当前线程的内存策略（get_mempolicy），返回 "default" / "bind 0,8" 等
static std::string current_policy() {
    const int bits = 8 * (int) sizeof(unsigned long);
    std::vector<unsigned long> mask(max_node()/bits + 2, 0);
    int mode = -1;
    if (syscall(SYS_get_mempolicy, &mode, mask.data(), (unsigned long) mask.size() * bits + 1, (void *) nullptr, 0) != 0) {
        return std::string("get_mempolicy failed: ") + strerror(errno);
    }
    std::string nodes;
    for (size_t w = 0; w < mask.size(); w++) {
        for (int b = 0; b < bits; b++) {
            if (mask[w] & (1ul << b)) nodes += (nodes.empty() ? "" : ",") + std::to_string((int) (w * bits + b));
        }
    }
    static const char * names[] = { "default", "preferred", "bind", "interleave", "local" };
    const char * name = mode >= 0 && mode < 5 ? names[mode] : "?";
    return std::string(name) + (nodes.empty() ? "" : " " + nodes);
}

// move_pages 抽样：返回 "0:48,8:16" 这样的节点分布；错误用 E<errno> 计（不足一页的块只查起点所在页）
static std::string page_nodes(void * ptr, size_t size, int samples = 64) {
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    const size_t n_pages = std::max<size_t>(size / page, 1);
    const size_t n = std::min<size_t>(n_pages, (size_t) samples);
    std::vector<void *> pages(n);
    std::vector<int>    status(n, -1);
    for (size_t i = 0; i < n; i++) pages[i] = (char *) ptr + (n_pages/n) * i * page;
    if (syscall(SYS_move_pages, 0, n, pages.data(), nullptr, status.data(), 0) != 0) {
        return std::string("move_pages:") + strerror(errno);
    }
    std::map<int, int> hist;
    for (int s : status) hist[s]++;
    std::string out;
    for (const auto & kv : hist) {
        char cell[32];
        if (kv.first >= 0) snprintf(cell, sizeof(cell), "%d:%d", kv.first, kv.second);
        else               snprintf(cell, sizeof(cell), "E%d:%d", -kv.first, kv.second);
        out += (out.empty() ? "" : ",") + std::string(cell);
    }
    return out;
}

// 在绑到 node 的 CPU 上的新线程里跑 fn（node < 0：不绑）
template <class F> static void run_on_node(int node, F && fn) {
    std::thread([&] {
        if (node >= 0) {
            cpu_set_t set;
            if (node_cpuset(node, set) && pthread_setaffinity_np(pthread_self(), sizeof set, &set) != 0) {
                fprintf(stderr, "pthread_setaffinity_np(node %d) failed: %s\n", node, strerror(errno));
            }
        }
        fn();
    }).join();
}

// ---------------------------------------------------------------- 读内核与计时

static __global__ void read_bw_kernel(const uint4 * __restrict__ src, size_t n_vec, unsigned int * __restrict__ sink) {
    const size_t tid    = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    unsigned int acc = 0;
    for (size_t i = tid; i < n_vec; i += stride) {
        const uint4 v = src[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    __shared__ unsigned int sdata[256];
    sdata[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] ^= sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicXor(sink, sdata[0]);
}

struct gpu_ctx {
    int dev = 0, node = -1, nsm = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    void * d_buf = nullptr;
    unsigned int * d_sink = nullptr;
};

static void gpu_init(gpu_ctx & g, int dev, size_t nbytes) {
    g.dev = dev;
    CK(cudaSetDevice(dev));
    g.node = gpu_node(dev);
    CK(cudaDeviceGetAttribute(&g.nsm, cudaDevAttrMultiProcessorCount, dev));
    CK(cudaStreamCreateWithFlags(&g.stream, cudaStreamNonBlocking));
    CK(cudaEventCreate(&g.ev0));
    CK(cudaEventCreate(&g.ev1));
    CK(cudaMalloc(&g.d_buf, nbytes));
    CK(cudaMalloc(&g.d_sink, sizeof(unsigned int)));
    CK(cudaMemset(g.d_sink, 0, sizeof(unsigned int)));
}

static double gbs(size_t nbytes, double ms) { return (double) nbytes / (ms * 1e6); }

static double median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// 直读：内核读 dev_ptr（主机内存的设备别名）
static double bench_read_ms(gpu_ctx & g, const void * dev_ptr, size_t nbytes, int reps) {
    CK(cudaSetDevice(g.dev));
    const size_t n_vec = nbytes / sizeof(uint4);
    const dim3 grid(g.nsm * 8);
    for (int i = 0; i < 2; i++) read_bw_kernel<<<grid, 256, 0, g.stream>>>((const uint4 *) dev_ptr, n_vec, g.d_sink);
    CK(cudaGetLastError());
    CK(cudaStreamSynchronize(g.stream));
    std::vector<float> ms(reps);
    for (int r = 0; r < reps; r++) {
        CK(cudaEventRecord(g.ev0, g.stream));
        read_bw_kernel<<<grid, 256, 0, g.stream>>>((const uint4 *) dev_ptr, n_vec, g.d_sink);
        CK(cudaGetLastError());
        CK(cudaEventRecord(g.ev1, g.stream));
        CK(cudaEventSynchronize(g.ev1));
        CK(cudaEventElapsedTime(&ms[r], g.ev0, g.ev1));
    }
    return median(ms);
}

// 拷贝：cudaMemcpyAsync 主机 → 显存
static double bench_copy_ms(gpu_ctx & g, const void * host_ptr, size_t nbytes, int reps) {
    CK(cudaSetDevice(g.dev));
    for (int i = 0; i < 2; i++) CK(cudaMemcpyAsync(g.d_buf, host_ptr, nbytes, cudaMemcpyHostToDevice, g.stream));
    CK(cudaStreamSynchronize(g.stream));
    std::vector<float> ms(reps);
    for (int r = 0; r < reps; r++) {
        CK(cudaEventRecord(g.ev0, g.stream));
        CK(cudaMemcpyAsync(g.d_buf, host_ptr, nbytes, cudaMemcpyHostToDevice, g.stream));
        CK(cudaEventRecord(g.ev1, g.stream));
        CK(cudaEventSynchronize(g.ev1));
        CK(cudaEventElapsedTime(&ms[r], g.ev0, g.ev1));
    }
    return median(ms);
}

// ---------------------------------------------------------------- 一块锁页内存的分配/填充/释放

enum class alloc_mode { hostalloc, mallochost, register_, policy, nobind };

static const char * mode_name(alloc_mode m) {
    switch (m) {
        case alloc_mode::hostalloc:  return "hostalloc";
        case alloc_mode::mallochost: return "mallochost";
        case alloc_mode::register_:  return "register";
        case alloc_mode::policy:     return "policy";
        case alloc_mode::nobind:     return "nobind";
    }
    return "?";
}

struct pinned_buf {
    alloc_mode mode = alloc_mode::hostalloc;
    void * host = nullptr;   // 主机地址
    void * dev  = nullptr;   // 设备别名
    size_t size = 0;
    std::string note;        // 页面分布等说明
    bool ok = false;
};

// 分配：alloc_node = 分配/注册线程绑到的节点（<0 不绑）；dev = 当前 GPU（cudaSetDevice 在分配线程里做）
static pinned_buf pinned_alloc(alloc_mode mode, int dev, size_t size, int alloc_node, int fill_node,
                               const std::vector<int> & nodes) {
    pinned_buf b;
    b.mode = mode;
    b.size = size;

    auto fill = [&](void * p) {
        // 填充线程绑到 fill_node，memset 使页面被该节点 CPU 最后写过（register 模式下也是首次触碰）
        run_on_node(fill_node, [&] { memset(p, 0x5a, size); });
    };

    cudaError_t err = cudaSuccess;
    switch (mode) {
        case alloc_mode::hostalloc:
        case alloc_mode::mallochost: {
            run_on_node(alloc_node, [&] {
                CK(cudaSetDevice(dev));
                err = mode == alloc_mode::hostalloc
                    ? cudaHostAlloc(&b.host, size, cudaHostAllocMapped | cudaHostAllocPortable)
                    : cudaMallocHost(&b.host, size);
            });
            if (err != cudaSuccess) { (void) cudaGetLastError(); b.note = std::string("alloc failed: ") + cudaGetErrorString(err); return b; }
            b.note = "fresh " + page_nodes(b.host, size);
            fill(b.host);
            b.note += " filled " + page_nodes(b.host, size);
            break;
        }
        case alloc_mode::register_: {
            void * p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (p == MAP_FAILED) { b.note = std::string("mmap failed: ") + strerror(errno); return b; }
            fill(p);   // 首次触碰决定页面所在节点
            b.note = "touched " + page_nodes(p, size);
            run_on_node(alloc_node, [&] {
                CK(cudaSetDevice(dev));
                err = cudaHostRegister(p, size, cudaHostRegisterMapped | cudaHostRegisterPortable);
            });
            if (err != cudaSuccess) { (void) cudaGetLastError(); munmap(p, size); b.note += std::string(" register failed: ") + cudaGetErrorString(err); return b; }
            b.host = p;
            b.note += " registered " + page_nodes(p, size);
            break;
        }
        case alloc_mode::policy: {
            // 我们 ggml-cuda.cu 的路径：set_mempolicy(MPOL_BIND alloc_node) + cudaHostAlloc + mbind(MPOL_MF_MOVE)。
            // 分配线程故意绑到另一个节点的 CPU，看驱动是按线程所在节点还是按 mempolicy 放页
            int other = -1;
            for (int n : nodes) if (n != alloc_node) { other = n; break; }
            std::string pol;
            run_on_node(other, [&] {
                CK(cudaSetDevice(dev));
                std::vector<unsigned long> mask;
                const unsigned long maxnode = nodemask_one(alloc_node, mask);
                if (syscall(SYS_set_mempolicy, MPOL_BIND, mask.data(), maxnode) != 0) {
                    pol = std::string("set_mempolicy failed: ") + strerror(errno) + "; ";
                }
                err = cudaHostAlloc(&b.host, size, cudaHostAllocMapped | cudaHostAllocPortable);
                syscall(SYS_set_mempolicy, MPOL_DEFAULT, (const unsigned long *) nullptr, 0ul);
            });
            if (err != cudaSuccess) { (void) cudaGetLastError(); b.note = pol + "alloc failed: " + cudaGetErrorString(err); return b; }
            b.note = pol + "thread@" + std::to_string(other) + " policy-fresh " + page_nodes(b.host, size);
            {
                std::vector<unsigned long> mask;
                const unsigned long maxnode = nodemask_one(alloc_node, mask);
                if (syscall(SYS_mbind, b.host, size, MPOL_BIND, mask.data(), maxnode, MPOL_MF_MOVE) != 0) {
                    b.note += std::string(" mbind failed: ") + strerror(errno);
                }
            }
            b.note += " mbind " + page_nodes(b.host, size);
            fill(b.host);
            b.note += " filled " + page_nodes(b.host, size);
            break;
        }
        case alloc_mode::nobind: {
            // moe-meas-0 的测法：主线程分配、主线程填充，节点由外部 numactl 决定
            CK(cudaSetDevice(dev));
            err = cudaHostAlloc(&b.host, size, cudaHostAllocMapped | cudaHostAllocPortable);
            if (err != cudaSuccess) { (void) cudaGetLastError(); b.note = std::string("alloc failed: ") + cudaGetErrorString(err); return b; }
            memset(b.host, 0x5a, size);
            b.note = "main-thread@cpu" + std::to_string(sched_getcpu()) + " " + page_nodes(b.host, size);
            break;
        }
    }

    CK(cudaSetDevice(dev));
    err = cudaHostGetDevicePointer(&b.dev, b.host, 0);
    if (err != cudaSuccess) { (void) cudaGetLastError(); b.note += std::string(" no device pointer: ") + cudaGetErrorString(err); return b; }
    b.ok = true;
    return b;
}

static void pinned_free(pinned_buf & b) {
    if (b.host == nullptr) return;
    if (b.mode == alloc_mode::register_) {
        CK(cudaHostUnregister(b.host));
        munmap(b.host, b.size);
    } else {
        CK(cudaFreeHost(b.host));
    }
    b.host = nullptr;
}

// ---------------------------------------------------------------- 主程序

static std::vector<int> parse_int_list(const char * s) { return parse_list(s); }

int main(int argc, char ** argv) {
    size_t mib = 1024;
    int reps = 5;
    std::vector<int> gpus;
    std::vector<alloc_mode> modes = { alloc_mode::hostalloc, alloc_mode::mallochost, alloc_mode::register_, alloc_mode::policy, alloc_mode::nobind };
    bool matrix = true, concurrent = false, small = false;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(1); } return argv[++i]; };
        if (a == "--mib") mib = strtoull(next(), nullptr, 10);
        else if (a == "--reps") reps = atoi(next());
        else if (a == "--gpus") gpus = parse_int_list(next());
        else if (a == "--modes") {
            modes.clear();
            char tmp[256]; snprintf(tmp, sizeof(tmp), "%s", next());
            for (char * tok = strtok(tmp, ","); tok; tok = strtok(nullptr, ",")) {
                const std::string m = tok;
                if      (m == "hostalloc")  modes.push_back(alloc_mode::hostalloc);
                else if (m == "mallochost") modes.push_back(alloc_mode::mallochost);
                else if (m == "register")   modes.push_back(alloc_mode::register_);
                else if (m == "policy")     modes.push_back(alloc_mode::policy);
                else if (m == "nobind")     modes.push_back(alloc_mode::nobind);
                else { fprintf(stderr, "unknown mode %s\n", tok); return 1; }
            }
        }
        else if (a == "--no-matrix") matrix = false;
        else if (a == "--concurrent") concurrent = true;
        else if (a == "--small") small = true;
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 1; }
    }

    const size_t nbytes = mib << 20;
    int n_dev = 0;
    CK(cudaGetDeviceCount(&n_dev));
    if (gpus.empty()) for (int d = 0; d < n_dev; d++) gpus.push_back(d);

    const std::vector<int> nodes = host_nodes();
    printf("bench-pinned-read: %zu MiB, %d reps, GPUs:", mib, reps);
    for (int d : gpus) printf(" %d", d);
    printf(", host NUMA nodes:");
    for (int n : nodes) printf(" %d", n);
    printf("\nmain thread: cpu %d (node %d), mempolicy: %s\n", sched_getcpu(), node_of_cpu(sched_getcpu(), nodes), current_policy().c_str());

    std::vector<gpu_ctx> ctx(n_dev);
    for (int d : gpus) {
        gpu_init(ctx[d], d, nbytes);
        cudaDeviceProp prop;
        CK(cudaGetDeviceProperties(&prop, d));
        printf("GPU %d: %s, %d SMs, NUMA node %d\n", d, prop.name, ctx[d].nsm, ctx[d].node);
    }
    fflush(stdout);

    if (matrix) {
        printf("\n%-4s %-5s %-10s %-6s %-5s %9s %9s  %s\n", "gpu", "gnode", "mode", "alloc", "fill", "read", "memcpy", "pages (move_pages sample of 64; fresh = right after allocation)");
        for (int d : gpus) {
            gpu_ctx & g = ctx[d];
            for (alloc_mode m : modes) {
                // nobind：只有一行（节点由外部 numactl 决定）
                std::vector<std::pair<int, int>> combos;
                if (m == alloc_mode::nobind) {
                    combos.push_back({ -1, -1 });
                } else {
                    for (int a : nodes) for (int f : nodes) combos.push_back({ a, f });
                }
                for (const auto & c : combos) {
                    pinned_buf b = pinned_alloc(m, d, nbytes, c.first, c.second, nodes);
                    char acol[8], fcol[8];
                    snprintf(acol, sizeof(acol), "%d", c.first);
                    snprintf(fcol, sizeof(fcol), "%d", c.second);
                    if (!b.ok) {
                        printf("%-4d %-5d %-10s %-6s %-5s %9s %9s  %s\n", d, g.node, mode_name(m),
                               c.first < 0 ? "-" : acol, c.second < 0 ? "-" : fcol, "-", "-", b.note.c_str());
                        pinned_free(b);
                        fflush(stdout);
                        continue;
                    }
                    const double t_read = bench_read_ms(g, b.dev, nbytes, reps);
                    const double t_copy = bench_copy_ms(g, b.host, nbytes, reps);
                    printf("%-4d %-5d %-10s %-6s %-5s %9.2f %9.2f  %s\n", d, g.node, mode_name(m),
                           c.first < 0 ? "-" : acol, c.second < 0 ? "-" : fcol,
                           gbs(nbytes, t_read), gbs(nbytes, t_copy), b.note.c_str());
                    fflush(stdout);
                    pinned_free(b);
                }
            }
        }
    }

    if (concurrent) {
        // 所有 GPU 同时读：三种放置
        struct placement { const char * name; int node; }; // node < 0：每卡本地
        std::vector<placement> pls = { { "local(alloc+fill on GPU node)", -1 } };
        for (int n : nodes) {
            static char names[8][48];
            static int k = 0;
            snprintf(names[k % 8], sizeof(names[0]), "all on node %d", n);
            pls.push_back({ names[k % 8], n });
            k++;
        }
        for (const placement & pl : pls) {
            std::vector<pinned_buf> bufs(n_dev);
            for (int d : gpus) {
                const int node = pl.node < 0 ? ctx[d].node : pl.node;
                bufs[d] = pinned_alloc(alloc_mode::hostalloc, d, nbytes, node, node, nodes);
                if (!bufs[d].ok) { fprintf(stderr, "GPU %d: %s\n", d, bufs[d].note.c_str()); return 1; }
            }
            std::atomic<int> ready(0);
            std::atomic<bool> go(false);
            std::vector<float> ms(n_dev, 0.0f);
            std::vector<std::thread> threads;
            for (int d : gpus) {
                threads.emplace_back([&, d] {
                    gpu_ctx & g = ctx[d];
                    CK(cudaSetDevice(d));
                    const size_t n_vec = nbytes / sizeof(uint4);
                    const dim3 grid(g.nsm * 8);
                    for (int i = 0; i < 2; i++) read_bw_kernel<<<grid, 256, 0, g.stream>>>((const uint4 *) bufs[d].dev, n_vec, g.d_sink);
                    CK(cudaGetLastError());
                    CK(cudaStreamSynchronize(g.stream));
                    ready++;
                    while (!go.load()) {}
                    CK(cudaEventRecord(g.ev0, g.stream));
                    for (int r = 0; r < reps; r++) read_bw_kernel<<<grid, 256, 0, g.stream>>>((const uint4 *) bufs[d].dev, n_vec, g.d_sink);
                    CK(cudaGetLastError());
                    CK(cudaEventRecord(g.ev1, g.stream));
                    CK(cudaEventSynchronize(g.ev1));
                    CK(cudaEventElapsedTime(&ms[d], g.ev0, g.ev1));
                });
            }
            while (ready.load() < (int) gpus.size()) {}
            go = true;
            for (auto & t : threads) t.join();
            double t_max = 0.0;
            printf("\nconcurrent read, %s:\n", pl.name);
            for (int d : gpus) {
                t_max = std::max(t_max, (double) ms[d]);
                printf("  GPU %d (node %d, pages %s): %7.2f GB/s\n", d, ctx[d].node, bufs[d].note.c_str(), gbs(nbytes * reps, ms[d]));
            }
            printf("  total: %7.2f GB/s (slowest GPU %.1f ms)\n", gbs(nbytes * reps * gpus.size(), t_max), t_max);
            fflush(stdout);
            for (int d : gpus) pinned_free(bufs[d]);
        }
    }

    if (small) {
        // 小块放置：分配线程绑到“另一”节点、set_mempolicy(MPOL_BIND 目标节点) 下连续分配不同大小的块，
        // 看页面落在哪个节点（策略是否足够）、起点页内偏移（驱动是否从共享的 2 MiB 映射里切小块）、
        // mbind(MPOL_MF_MOVE) 对这些块的返回值（非页对齐起点应 EINVAL）。只用第一块 GPU 的上下文。
        const int dev = gpus.front();
        const size_t page = (size_t) sysconf(_SC_PAGESIZE);
        const std::vector<size_t> sizes = { 20, 32, 64 << 10, 512 << 10, 900 << 10, 1 << 20, 2 << 20, 4 << 20, 32 << 20, 128 << 20 };
        printf("\nsmall allocations: cudaHostAlloc(Mapped|Portable) under set_mempolicy(MPOL_BIND target) from a thread on the other node; page %zu KiB\n", page >> 10);
        printf("%-6s %-10s %-18s %-10s %-24s %s\n", "target", "bytes", "address", "page-off", "pages (move_pages)", "mbind(MPOL_MF_MOVE)");
        for (int target : nodes) {
            int other = -1;
            for (int n : nodes) if (n != target) { other = n; break; }
            std::vector<void *> ptrs(sizes.size(), nullptr);
            run_on_node(other, [&] {
                CK(cudaSetDevice(dev));
                std::vector<unsigned long> mask;
                const unsigned long maxnode = nodemask_one(target, mask);
                if (syscall(SYS_set_mempolicy, MPOL_BIND, mask.data(), maxnode) != 0) {
                    fprintf(stderr, "set_mempolicy(%d) failed: %s\n", target, strerror(errno));
                }
                for (size_t i = 0; i < sizes.size(); i++) {
                    if (cudaHostAlloc(&ptrs[i], sizes[i], cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
                        (void) cudaGetLastError();
                        ptrs[i] = nullptr;
                    }
                }
                syscall(SYS_set_mempolicy, MPOL_DEFAULT, (const unsigned long *) nullptr, 0ul);
            });
            for (size_t i = 0; i < sizes.size(); i++) {
                if (ptrs[i] == nullptr) {
                    printf("%-6d %-10zu %-18s\n", target, sizes[i], "alloc failed");
                    continue;
                }
                std::vector<unsigned long> mask;
                const unsigned long maxnode = nodemask_one(target, mask);
                const long rc = syscall(SYS_mbind, ptrs[i], sizes[i], MPOL_BIND, mask.data(), maxnode, MPOL_MF_MOVE);
                const int  e  = errno;
                char mb[64];
                if (rc == 0) snprintf(mb, sizeof(mb), "ok");
                else         snprintf(mb, sizeof(mb), "rc %ld errno %d (%s)", rc, e, strerror(e));
                printf("%-6d %-10zu %-18p %-10zu %-24s %s\n", target, sizes[i], ptrs[i],
                       (size_t) ((uintptr_t) ptrs[i] % page), page_nodes(ptrs[i], sizes[i], 16).c_str(), mb);
            }
            CK(cudaSetDevice(dev));
            for (void * p : ptrs) if (p != nullptr) CK(cudaFreeHost(p));
        }
        fflush(stdout);
    }

    for (int d : gpus) {
        CK(cudaSetDevice(d));
        CK(cudaFree(ctx[d].d_buf));
        CK(cudaFree(ctx[d].d_sink));
        CK(cudaEventDestroy(ctx[d].ev0));
        CK(cudaEventDestroy(ctx[d].ev1));
        CK(cudaStreamDestroy(ctx[d].stream));
    }
    printf("BENCH_DONE\n");
    return 0;
}

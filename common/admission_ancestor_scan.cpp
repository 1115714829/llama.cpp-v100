#include "admission_ancestor_scan.h"
#include <cerrno>
#include <cstring>
#include <limits>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace admission_ancestor_scan {
#ifdef COMMON_ADMISSION_TESTING
static thread_local between_passes_hook pass_hook = nullptr;
static thread_local void * pass_context = nullptr;
void test_between_passes(between_passes_hook hook, void * context) noexcept {
    pass_hook = hook; pass_context = context;
}
#endif
namespace {
struct fd_owner {
    int fd;
    explicit fd_owner(int f = -1) noexcept : fd(f) {}
    ~fd_owner() { if (fd >= 0) close(fd); }
    fd_owner(const fd_owner &) = delete;
};
bool integer(const char * p, uint64_t & n) noexcept {
    n = 0;
    if (*p < '0' || *p > '9') return false;
    do {
        const unsigned digit = unsigned(*p++ - '0');
        if (n > (UINT64_MAX - digit)/10) return false;
        n = n*10 + digit;
    } while (*p >= '0' && *p <= '9');
    while (*p == '\n' || *p == ' ' || *p == '\t') ++p;
    return *p == 0;
}
error read_scalar(int dir, const char * name, uint64_t & number, bool allow_max, bool & unlimited) noexcept {
    fd_owner file(openat(dir, name, O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    if (file.fd < 0) return error::inaccessible;
    char buffer[96]; size_t count = 0;
    for (;;) {
        const ssize_t n = read(file.fd, buffer + count, sizeof(buffer)-1-count);
        if (n < 0) { if (errno == EINTR) continue; return error::inaccessible; }
        if (!n) break;
        count += size_t(n);
        if (count == sizeof(buffer)-1) return error::malformed;
    }
    if (memchr(buffer, 0, count)) return error::malformed;
    buffer[count] = 0;
    unlimited = allow_max && (!strcmp(buffer,"max\n") || !strcmp(buffer,"max"));
    if (unlimited) { number = 0; return error::ok; }
    return integer(buffer, number) ? error::ok : error::malformed;
}
bool canonical_components(const char * path, bool root) noexcept {
    const size_t len = strlen(path);
    if (!len) return !root;
    if (root && len == 1 && path[0] == '/') return true;
    if (path[len-1] == '/' || strstr(path, "//")) return false;
    const char * p = path + (path[0] == '/' ? 1 : 0);
    while (*p) {
        const char * slash = strchr(p, '/');
        const size_t n = slash ? size_t(slash-p) : strlen(p);
        if (!n || n > 255 || (n==1 && p[0]=='.') || (n==2 && p[0]=='.' && p[1]=='.')) return false;
        if (!slash) break;
        p = slash+1;
    }
    return true;
}
int open_components(int start, const char * path) noexcept {
    int current = fcntl(start,F_DUPFD_CLOEXEC,0);
    if (current < 0) return -1;
    const char * p = path;
    while (*p) {
        while (*p == '/') ++p;
        if (!*p) break;
        char part[256]; size_t n = 0;
        while (*p && *p != '/') {
            if (n == sizeof(part)-1) { close(current); errno=ENAMETOOLONG; return -1; }
            part[n++] = *p++;
        }
        part[n]=0;
        if (!strcmp(part,".") || !strcmp(part,"..")) { close(current); errno=EINVAL; return -1; }
        const int next = openat(current, part, O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW);
        close(current); current = next;
        if (current < 0) return -1;
    }
    return current;
}
error read_level(int dir, unsigned version, bool root, level & out) noexcept {
    struct stat st;
    if (fstat(dir,&st)) return error::inaccessible;
    out = {}; out.device=st.st_dev; out.inode=st.st_ino; out.version=version; out.visible_root=root;
    bool unlimited=false, dummy=false;
    error e=read_scalar(dir, version==2?"memory.max":"memory.limit_in_bytes",out.limit,version==2,unlimited);
    if (e!=error::ok) return e;
    e=read_scalar(dir, version==2?"memory.current":"memory.usage_in_bytes",out.current,false,dummy);
    if (e!=error::ok) return e;
    if (version==1) {
        uint64_t hierarchy=0;
        e=read_scalar(dir,"memory.use_hierarchy",hierarchy,false,dummy);
        if (e!=error::ok || hierarchy!=1) return error::hierarchy_unknown;
        unlimited=out.limit >= (uint64_t(1)<<62);
    }
    out.finite=!unlimited;

    out.hard_usage=version==2 || out.finite;
    return error::ok;
}
}
error scan(const char * root, const char * leaf, unsigned version, snapshot & out) noexcept {
    out.count=0; out.status=error::invalid_path;
    if (!root || !leaf || !*root || *leaf=='/' || (version!=1 && version!=2)) return out.status;
    if (strnlen(root,max_path)==max_path || strnlen(leaf,max_path)==max_path) return out.status;
    if (!canonical_components(root,true) || !canonical_components(leaf,false)) return out.status;
    fd_owner anchor(open(*root=='/'?"/":".",O_RDONLY|O_DIRECTORY|O_CLOEXEC));
    if (anchor.fd<0) return out.status=error::inaccessible;
    fd_owner root_fd(open_components(anchor.fd,root));
    if (root_fd.fd<0) return out.status=error::inaccessible;
    return scan_fd(root_fd.fd, leaf, version, out);
}
error scan_fd(int controller_fd, const char * leaf, unsigned version, snapshot & out) noexcept {
    out.count=0; out.status=error::invalid_path;
    if (controller_fd < 0 || !leaf || *leaf=='/' || (version!=1 && version!=2) ||
        strnlen(leaf,max_path)==max_path || !canonical_components(leaf,false)) return out.status;
    fd_owner root_fd(fcntl(controller_fd,F_DUPFD_CLOEXEC,0));
    struct stat root_stat;
    if (root_fd.fd<0 || fstat(root_fd.fd,&root_stat) || !S_ISDIR(root_stat.st_mode)) return out.status=error::inaccessible;
    char relative[max_path]; memcpy(relative,leaf,strlen(leaf)+1);
    for (;;) {
        if (out.count==max_levels) return out.status=error::capacity;
        fd_owner dir(open_components(root_fd.fd,relative));
        if (dir.fd<0) return out.status=error::inaccessible;
        auto e=read_level(dir.fd,version,!*relative,out.levels[out.count]);
        if (e!=error::ok) return out.status=e;
        ++out.count;
        if (!*relative) break;
        char * slash=strrchr(relative,'/');
        if (slash) *slash=0; else *relative=0;
    }
#ifdef COMMON_ADMISSION_TESTING
    if (pass_hook) pass_hook(pass_context);
#endif

    memcpy(relative,leaf,strlen(leaf)+1);
    for (size_t i=0;i<out.count;++i) {
        fd_owner dir(open_components(root_fd.fd,relative)); level again{};
        if (dir.fd<0) return out.status=error::changed;
        const auto e=read_level(dir.fd,version,!*relative,again);
        if(e!=error::ok) return out.status=e;
        auto & old=out.levels[i];
        if(old.device!=again.device || old.inode!=again.inode || old.limit!=again.limit || old.finite!=again.finite || old.hard_usage!=again.hard_usage)
            return out.status=error::changed;
        if(again.current>old.current) old.current=again.current;
        char * slash=strrchr(relative,'/'); if(slash)*slash=0;else *relative=0;
    }
    return out.status=error::ok;
}
}

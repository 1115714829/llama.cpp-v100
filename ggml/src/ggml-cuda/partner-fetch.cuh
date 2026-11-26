#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include <cuda/atomic>

struct ggml_partner_fetch_site {
    const int32_t * plan;
    const int * count;
    const int64_t * host_offset;
    const char * mapped_host;
    char * peer_stage;
    size_t bytes, stride;
    unsigned cap;
};

static __device__ void ggml_partner_copy_odd(ggml_partner_fetch_site s) {
    const int count = *s.count;
    if (count <= 1) return;
    const size_t slots = size_t(count)/2;
    const size_t nt = size_t(gridDim.x)*blockDim.x;
    const size_t lane = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    for (size_t pair=0; pair<slots && 2*pair+1<s.cap; ++pair) {
        const size_t slot = 2*pair+1;
        const char * src = s.mapped_host+s.host_offset[s.plan[slot]];
        char * dst = s.peer_stage+slot*s.stride;
        const bool aligned = ((uintptr_t(src)|uintptr_t(dst)) & 15u)==0;
        const size_t vec = aligned ? s.bytes/16 : 0;
        for (size_t i=lane; i<vec; i+=nt) reinterpret_cast<uint4 *>(dst)[i]=reinterpret_cast<const uint4 *>(src)[i];
        for (size_t i=16*vec+lane; i<s.bytes; i+=nt) dst[i]=src[i];
        for (size_t i=s.bytes+lane; i<s.stride; i+=nt) dst[i]=0;
    }
    __threadfence_system();
}

struct partner_mail { ggml_partner_fetch_site site; unsigned long long bytes; };
using partner_flag = cuda::atomic_ref<unsigned,cuda::thread_scope_system>;
static __device__ void partner_wait(unsigned &flag,unsigned value) {
    const auto start=clock64();
    while(partner_flag(flag).load(cuda::memory_order_acquire)!=value) {
        if(clock64()-start>3000000000ULL) asm volatile("trap;");
        __nanosleep(200);
    }
}
static __global__ void ggml_partner_window(partner_mail *mail,int n,unsigned *arrive,unsigned *ready,unsigned *done,unsigned epoch) {
    for(int i=0;i<n;++i) {
        __shared__ unsigned cmd;
        if(!threadIdx.x){const auto start=clock64();unsigned v;
            do { v=partner_flag(ready[i]).load(cuda::memory_order_acquire);
                if(clock64()-start>3000000000ULL) asm volatile("trap;");
                if((v>>1)<epoch) __nanosleep(200);
            }while((v>>1)<epoch);
            cmd=v;
        }
        __syncthreads();

        const bool work=cmd==((epoch<<1)|1u); __syncthreads(); if(!work) continue;
        auto &m=mail[i]; ggml_partner_copy_odd(m.site);
        __threadfence_system(); __syncthreads();
        if(!threadIdx.x && atomicInc(arrive+i,gridDim.x-1)==gridDim.x-1){
            m.bytes+=(unsigned long long)(*m.site.count/2)*m.site.bytes;
            partner_flag(done[i]).store(epoch,cuda::memory_order_release);
        }
        __syncthreads();
    }
}

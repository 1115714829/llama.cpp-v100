#pragma once
#include <cstddef>
#include <cstdint>
namespace admission_numa {
constexpr unsigned node_bits=4096;
constexpr unsigned words=node_bits/(8*sizeof(unsigned long));
struct mask { unsigned long bits[words]{}; };
enum class code { ok, io, malformed, capacity, unsupported, permission, syscall_failed, changed, mismatch, empty };
struct status { code error; int system_error; const char * stage; };

struct operations {
    void * context;
    status (*read)(void *, const char *, char *, size_t, size_t &) noexcept;
    status (*allowed)(void *, mask &) noexcept;
    status (*bind)(void *, void *, size_t, const mask &) noexcept;
    status (*policy)(void *, void *, int &, mask &) noexcept;
};
status apply(void * address,size_t bytes,const operations & ops) noexcept;
status apply_system(void * address,size_t bytes) noexcept;
status parse_list(const char * text,size_t length,mask & out,bool allow_empty) noexcept;
status parse_total(const char * text,size_t length,unsigned node,uint64_t & bytes) noexcept;
#ifdef COMMON_ADMISSION_OWNER_TESTING
status test_read_file(const char *,char *,size_t,size_t &,size_t,unsigned,int) noexcept;
#endif
}

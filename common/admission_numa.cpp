#include "admission_numa.h"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/mempolicy.h>
namespace admission_numa {
namespace {
status fail(code c,const char * stage,int e=0) noexcept {return {c,e,stage};}
bool bit(const mask & m,unsigned n) noexcept {return m.bits[n/(8*sizeof(unsigned long))]&(1ul<<(n%(8*sizeof(unsigned long))));}
void set(mask & m,unsigned n) noexcept {m.bits[n/(8*sizeof(unsigned long))]|=1ul<<(n%(8*sizeof(unsigned long)));}
bool decimal(const char * & p,const char * end,uint64_t & n) noexcept {
 n=0;const char * start=p;
 while(p<end&&*p>='0'&&*p<='9'){unsigned d=unsigned(*p++-'0');if(n>(UINT64_MAX-d)/10)return false;n=n*10+d;}
 return p!=start;
}
status sys_error(const char * stage) noexcept {
 return fail(errno==ENOSYS?code::unsupported:((errno==EPERM||errno==EACCES)?code::permission:(errno==EINVAL?code::capacity:code::syscall_failed)),stage,errno);
}
#ifdef COMMON_ADMISSION_OWNER_TESTING
thread_local size_t test_chunk = SIZE_MAX;
thread_local unsigned test_interrupts = 0;
thread_local int test_errno = 0;
#endif
ssize_t read_once(int fd, void * data, size_t count) noexcept {
#ifdef COMMON_ADMISSION_OWNER_TESTING
    if (test_interrupts) { --test_interrupts; errno=EINTR; return -1; }
    if (test_errno) { errno=test_errno; return -1; }
    if (count>test_chunk) count=test_chunk;
#endif
    return read(fd,data,count);
}
status read_file(void *,const char * path,char * data,size_t cap,size_t & n) noexcept {
    n=0;
    int fd=open(path,O_RDONLY|O_CLOEXEC|O_NOFOLLOW);
    if(fd<0) return sys_error("read-open");
    while(n<cap) {
        const ssize_t got=read_once(fd,data+n,cap-n);
        if(got<0) {
            if(errno==EINTR) continue;
            const int e=errno;
            close(fd);
            return fail(code::io,"read",e);
        }
        if(!got) break;
        n+=size_t(got);
    }
    if(n==cap) {
        char c;
        ssize_t got;
        do { got=read_once(fd,&c,1); } while(got<0 && errno==EINTR);
        const int e=got<0 ? errno : 0;
        if(got!=0) {
            close(fd);
            return fail(got<0 ? code::io : code::capacity,"read-capacity",e);
        }
    }
    close(fd);
    if(memchr(data,0,n)) return fail(code::malformed,"read-nul");
    return fail(code::ok,"read");
}
status allowed(void *,mask & m) noexcept {
 if(syscall(SYS_get_mempolicy,nullptr,m.bits,node_bits,nullptr,MPOL_F_MEMS_ALLOWED))return sys_error("allowed");
  return fail(code::ok,"allowed");
}
status bind(void *,void * p,size_t n,const mask & m) noexcept {
 if(syscall(SYS_mbind,p,n,MPOL_BIND|MPOL_F_STATIC_NODES,m.bits,node_bits,0))return sys_error("mbind");
  return fail(code::ok,"mbind");
}
status policy(void *,void * p,int & mode,mask & m) noexcept {
 if(syscall(SYS_get_mempolicy,&mode,m.bits,node_bits,p,MPOL_F_ADDR))return sys_error("policy-readback");
  return fail(code::ok,"policy-readback");
}
}
#ifdef COMMON_ADMISSION_OWNER_TESTING
status test_read_file(const char * path,char * data,size_t cap,size_t & n,size_t chunk,unsigned interrupts,int error) noexcept {
    test_chunk=chunk ? chunk : SIZE_MAX;
    test_interrupts=interrupts;
    test_errno=error;
    const auto result=read_file(nullptr,path,data,cap,n);
    test_chunk=SIZE_MAX; test_interrupts=0; test_errno=0;
    return result;
}
#endif
status parse_list(const char * text,size_t n,mask & out,bool allow_empty) noexcept {
 out={};if(!text||memchr(text,0,n))return fail(code::malformed,"list");
 while(n&&(text[n-1]=='\n'||text[n-1]==' '||text[n-1]=='\t'))--n;
 if(!n)return fail(allow_empty?code::ok:code::malformed,"list-empty");
 const char * p=text,*end=text+n;uint64_t previous=0;bool have=false;
 while(p<end){uint64_t first,last;if(!decimal(p,end,first))return fail(code::malformed,"list-number");last=first;
  if(p<end&&*p=='-'){++p;if(!decimal(p,end,last))return fail(code::malformed,"list-range");}
  if(last<first||(have&&first<=previous))return fail(code::malformed,"list-order");
  if(last>=node_bits)return fail(code::capacity,"list-capacity");
  for(uint64_t x=first;x<=last;++x)set(out,unsigned(x));have=true;previous=last;
  if(p==end)break;if(*p++!=','||p==end)return fail(code::malformed,"list-separator");
 }
 return fail(code::ok,"list");
}
status parse_total(const char * text,size_t n,unsigned node,uint64_t & bytes) noexcept {
 bytes=0;if(!text||memchr(text,0,n))return fail(code::malformed,"memtotal");
 char prefix[64];int length=snprintf(prefix,sizeof(prefix),"Node %u MemTotal:",node);bool found=false;
 const char * p=text,*end=text+n;
 while(p<end){const char * e=static_cast<const char *>(memchr(p,'\n',size_t(end-p)));if(!e)return fail(code::malformed,"memtotal-truncated");
  if(size_t(e-p)>=size_t(length)&&!memcmp(p,prefix,size_t(length))){if(found)return fail(code::malformed,"memtotal-duplicate");found=true;
   const char * q=p+length;while(q<e&&(*q==' '||*q=='\t'))++q;uint64_t value;
   if(!decimal(q,e,value))return fail(code::malformed,"memtotal-number");
   if(q==e||(*q!=' '&&*q!='\t'))return fail(code::malformed,"memtotal-unit");while(q<e&&(*q==' '||*q=='\t'))++q;
   if(e-q!=2||memcmp(q,"kB",2))return fail(code::malformed,"memtotal-unit");
   if(value>UINT64_MAX/1024)return fail(code::capacity,"memtotal-overflow");bytes=value*1024;
  }p=e+1;
 }return fail(found?code::ok:code::malformed,"memtotal");
}
status apply(void * p,size_t bytes,const operations & ops) noexcept {
 if(!p||!bytes||!ops.read||!ops.allowed||!ops.bind||!ops.policy)return fail(code::malformed,"arguments");
 mask possible{},permitted{},chosen{},again{},reported{};char data[16384],path[160];size_t n=0;
 auto s=ops.read(ops.context,"/sys/devices/system/node/possible",data,sizeof(data),n);if(s.error!=code::ok)return s;
 s=parse_list(data,n,possible,false);if(s.error!=code::ok)return s;
 s=ops.allowed(ops.context,permitted);if(s.error!=code::ok)return s;bool any=false;
 for(unsigned node=0;node<node_bits;++node){if(!bit(permitted,node))continue;if(!bit(possible,node))return fail(code::changed,"allowed-not-possible");
  snprintf(path,sizeof(path),"/sys/devices/system/node/node%u/cpulist",node);
  s=ops.read(ops.context,path,data,sizeof(data),n);if(s.error!=code::ok)return s;
  mask cpus{};s=parse_list(data,n,cpus,true);if(s.error!=code::ok)return s;bool cpu=false;for(auto w:cpus.bits)cpu|=w!=0;if(!cpu)continue;
  snprintf(path,sizeof(path),"/sys/devices/system/node/node%u/meminfo",node);
  s=ops.read(ops.context,path,data,sizeof(data),n);if(s.error!=code::ok)return s;uint64_t total;
  s=parse_total(data,n,node,total);if(s.error!=code::ok)return s;if(!total)continue;set(chosen,node);any=true;
 }
 if(!any)return fail(code::empty,"cpu-dram-intersection");
 s=ops.bind(ops.context,p,bytes,chosen);if(s.error!=code::ok)return s;
 int mode=0;s=ops.policy(ops.context,p,mode,reported);if(s.error!=code::ok)return s;
 if((mode&~MPOL_F_STATIC_NODES)!=MPOL_BIND||memcmp(chosen.bits,reported.bits,sizeof(chosen)))return fail(code::mismatch,"policy-readback");
 s=ops.allowed(ops.context,again);if(s.error!=code::ok)return s;
 if(memcmp(permitted.bits,again.bits,sizeof(permitted)))return fail(code::changed,"allowed-changed");
 return fail(code::ok,"complete");
}
status apply_system(void * p,size_t bytes) noexcept {return apply(p,bytes,{nullptr,read_file,allowed,bind,policy});}
}

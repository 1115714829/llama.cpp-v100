#include "admission_cgroup_observe.h"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/vfs.h>
#include <sys/sysmacros.h>
#include <linux/magic.h>
#include <sys/syscall.h>

namespace admission_cgroup_observe {
namespace {
#ifdef COMMON_ADMISSION_TESTING
thread_local size_t read_cap = SIZE_MAX;
thread_local bool eintr_once = false;
thread_local test_hooks * hooks = nullptr;
#endif
struct fd_owner {
    int fd;
    explicit fd_owner(int value=-1) noexcept : fd(value) {}
    ~fd_owner() { if(fd>=0) close(fd); }
    fd_owner(const fd_owner &)=delete;
};
ssize_t observed_read(int fd, void * buffer, size_t count) noexcept {
#ifdef COMMON_ADMISSION_TESTING
    if(hooks && ++hooks->read_calls==hooks->interrupt_call){errno=EINTR;return -1;}
    if(eintr_once){eintr_once=false;errno=EINTR;return -1;}
    if(hooks && hooks->interrupted_reads){--hooks->interrupted_reads;errno=EINTR;return -1;}
    if(hooks && hooks->read_error){errno=hooks->read_error;return -1;}
    if(count>read_cap)count=read_cap;
    if(hooks && count>hooks->read_chunk)count=hooks->read_chunk;
#endif
    return read(fd,buffer,count);
}
int observed_stat(int fd,struct stat * st) noexcept {
    int rc=fstat(fd,st);
#ifdef COMMON_ADMISSION_TESTING
    if(!rc && hooks && hooks->stat_identity){
        unsigned long long dev=st->st_dev,ino=st->st_ino;
        hooks->stat_identity(fd,dev,ino);st->st_dev=dev;st->st_ino=ino;
    }
#endif
    return rc;
}
int observed_statfs(int fd,struct statfs * st) noexcept {
    int rc=fstatfs(fd,st);
#ifdef COMMON_ADMISSION_TESTING
    if(!rc && hooks && hooks->filesystem_type)st->f_type=hooks->filesystem_type(fd,st->f_type);
#endif
    return rc;
}
error read_all(int directory,const char * file,char * buffer,size_t capacity,size_t & used) noexcept {
    used=0;
    fd_owner f(openat(directory,file,O_RDONLY|O_CLOEXEC|O_NOFOLLOW));
    if(f.fd<0)return error::io;
    while(used<capacity) {
        ssize_t n=observed_read(f.fd,buffer+used,capacity-used);
        if(n<0){if(errno==EINTR)continue;return error::io;}
        if(!n)break;
        if(memchr(buffer+used,0,size_t(n)))return error::embedded_nul;
        used+=size_t(n);
    }
    if(used==capacity) {
        char extra; ssize_t n;
        do { n=observed_read(f.fd,&extra,1); } while(n<0&&errno==EINTR);
        if(n<0)return error::io;
        if(n)return error::capacity;
    }
    if(!used||buffer[used-1]!='\n')return error::truncated;
    return error::ok_local;
}
int directory_at(int root,const char * path) noexcept {
    int current=fcntl(root,F_DUPFD_CLOEXEC,0);
    if(current<0)return -1;
    const char * p=path;
    while(*p=='/')++p;
    while(*p) {
        char part[256];size_t n=0;
        while(*p&&*p!='/'){if(n==255){close(current);return -1;}part[n++]=*p++;}
        part[n]=0;
        if(!n||!strcmp(part,".")||!strcmp(part,"..")){close(current);return -1;}
        int next=openat(current,part,O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW);
        close(current);current=next;
        if(current<0)return -1;
        if(*p=='/')++p;
    }
    return current;
}
bool same(const struct stat & a,const struct stat & b) noexcept {return a.st_dev==b.st_dev&&a.st_ino==b.st_ino;}
bool mount_id(int proc,int fd,uint64_t & id) noexcept {
    char path[64],text[4096];size_t n=0;
#ifdef COMMON_ADMISSION_TESTING
    if(hooks && hooks->fdinfo_bytes) {
        const char * raw=hooks->fdinfo_bytes(fd,n);
        if(!raw || !n || n>=sizeof(text) || memchr(raw,0,n) || raw[n-1]!='\n')return false;
        memcpy(text,raw,n);
    } else
#endif
    {
        snprintf(path,sizeof(path),"fdinfo/%d",fd);
        if(read_all(proc,path,text,sizeof(text)-1,n)!=error::ok_local)return false;
    }
    text[n]=0;const char * p=text;bool found=false;
    while(*p) {
        const char * end=strchr(p,'\n');if(!end)return false;
        if(size_t(end-p)>=7&&!memcmp(p,"mnt_id:",7)) {
            if (found) {
                return false;
            }
            found = true;
            p += 7;
            while(p<end&&(*p==' '||*p=='\t'))++p;
            id=0;bool digit=false;
            while(p<end&&*p>='0'&&*p<='9') {digit=true;unsigned d=unsigned(*p++-'0');if(id>(UINT64_MAX-d)/10)return false;id=id*10+d;}
            if(!digit||p!=end)return false;
        }
        p=end+1;
    }
    return found;
}
bool controller(int proc,int fd,const admission_cgroup_resolver::result & map) noexcept {
    struct stat st;struct statfs fs;uint64_t id=0;
    if(observed_stat(fd,&st)||observed_statfs(fd,&fs)||!S_ISDIR(st.st_mode))return false;
    const long wanted=map.version==2?CGROUP2_SUPER_MAGIC:CGROUP_SUPER_MAGIC;
    return fs.f_type==wanted && uint64_t(major(st.st_dev))==map.device_major &&
        uint64_t(minor(st.st_dev))==map.device_minor && mount_id(proc,fd,id)&&id==map.mount_id;
}
}
#ifdef COMMON_ADMISSION_TESTING
void set_test_hooks(test_hooks * value) noexcept { hooks=value; }
error test_read(int directory,const char * name,char * buffer,size_t capacity,size_t & used,size_t max_read,bool inject_eintr) noexcept {
    read_cap=max_read?max_read:SIZE_MAX;eintr_once=inject_eintr;
    const auto status=read_all(directory,name,buffer,capacity,used);
    read_cap=SIZE_MAX;eintr_once=false;return status;
}
#endif
error observe(workspace & w,result & out) noexcept {
    out={};out.status=error::io;

    const char * proc_path="/proc";
    const char * fs_path="/";
#ifdef COMMON_ADMISSION_TESTING
    if(hooks && hooks->proc_root)proc_path=hooks->proc_root;
    if(hooks && hooks->filesystem_root)fs_path=hooks->filesystem_root;
#endif
    fd_owner proc_root(open(proc_path,O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW));
    struct statfs procfs;
    if(proc_root.fd<0||observed_statfs(proc_root.fd,&procfs)||procfs.f_type!=PROC_SUPER_MAGIC)return out.status;
    char task_path[96];
    snprintf(task_path,sizeof(task_path),"%ld/task/%ld",(long)getpid(),(long)syscall(SYS_gettid));
    fd_owner proc(directory_at(proc_root.fd,task_path));
    if(proc.fd<0)return out.status;

    fd_owner ns_mnt(openat(proc.fd,"ns/mnt",O_RDONLY|O_CLOEXEC));
    fd_owner ns_cg(openat(proc.fd,"ns/cgroup",O_RDONLY|O_CLOEXEC));
    struct stat mnt_before,cg_before;
    if(ns_mnt.fd<0||ns_cg.fd<0||fstat(ns_mnt.fd,&mnt_before)||fstat(ns_cg.fd,&cg_before))return out.status;
    size_t mn=0,cn=0;
    auto e=read_all(proc.fd,"mountinfo",w.mounts,sizeof(w.mounts),mn);
    if(e!=error::ok_local)return out.status=e;
    e=read_all(proc.fd,"cgroup",w.membership,sizeof(w.membership),cn);
    if(e!=error::ok_local)return out.status=e;
    if(admission_cgroup_resolver::resolve(w.mounts,mn,w.membership,cn,out.mapping)!=admission_cgroup_resolver::error::ok)return out.status=error::parse;
    fd_owner slash(open(fs_path,O_RDONLY|O_DIRECTORY|O_CLOEXEC));
    fd_owner root(directory_at(slash.fd,out.mapping.mountpoint));
    if(root.fd<0||!controller(proc.fd,root.fd,out.mapping))return out.status=error::identity;
    fd_owner leaf(directory_at(root.fd,out.mapping.relative_leaf));
    if(leaf.fd<0||!controller(proc.fd,leaf.fd,out.mapping))return out.status=error::identity;
    struct stat root_before,leaf_before;
    if(fstat(root.fd,&root_before)||fstat(leaf.fd,&leaf_before))return out.status;
    if(admission_ancestor_scan::scan_fd(root.fd,out.mapping.relative_leaf,out.mapping.version,out.constraints)!=admission_ancestor_scan::error::ok)return out.status=error::scanner_unknown;

    if(!out.constraints.count||out.constraints.levels[0].inode!=uint64_t(leaf_before.st_ino)||
       out.constraints.levels[0].device!=uint64_t(leaf_before.st_dev))return out.status=error::changed;
#ifdef COMMON_ADMISSION_TESTING
    if(hooks && hooks->after_scan)hooks->after_scan();
#endif
    size_t rn=0;
    e=read_all(proc.fd,"mountinfo",w.reread,sizeof(w.reread),rn);
    if(e!=error::ok_local)return out.status=e;
    if(rn!=mn||memcmp(w.mounts,w.reread,mn))return out.status=error::changed;
    e=read_all(proc.fd,"cgroup",w.reread,sizeof(w.reread),rn);
    if(e!=error::ok_local)return out.status=e;
    if(rn!=cn||memcmp(w.membership,w.reread,cn))return out.status=error::changed;
    struct stat mnt_after,cg_after,root_after,leaf_after;
    if(fstatat(proc.fd,"ns/mnt",&mnt_after,0)||fstatat(proc.fd,"ns/cgroup",&cg_after,0)||
       fstat(root.fd,&root_after)||fstat(leaf.fd,&leaf_after))return out.status;
    if(!same(mnt_before,mnt_after)||!same(cg_before,cg_after)||!same(root_before,root_after)||!same(leaf_before,leaf_after))return out.status=error::changed;
    if(!controller(proc.fd,root.fd,out.mapping)||!controller(proc.fd,leaf.fd,out.mapping))return out.status=error::changed;

    fd_owner reopened(directory_at(slash.fd,out.mapping.mountpoint));struct stat reopened_stat;
    if(reopened.fd<0||fstat(reopened.fd,&reopened_stat)||!same(root_before,reopened_stat)||!controller(proc.fd,reopened.fd,out.mapping))return out.status=error::changed;
    fd_owner reopened_leaf(directory_at(reopened.fd,out.mapping.relative_leaf));
    struct stat reopened_leaf_stat;
    if(reopened_leaf.fd<0 || fstat(reopened_leaf.fd,&reopened_leaf_stat) ||
       !same(leaf_before,reopened_leaf_stat) || !controller(proc.fd,reopened_leaf.fd,out.mapping))
        return out.status=error::changed;
    out.global_ancestry_proven=false;
    return out.status=error::ok_local;
}
}

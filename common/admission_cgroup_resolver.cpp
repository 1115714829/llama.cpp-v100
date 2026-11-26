#include "admission_cgroup_resolver.h"
#include <cstring>
#include <cstdint>

namespace admission_cgroup_resolver {
namespace {
bool number(const char * p, uint64_t & n) noexcept {
    n = 0;
    if (!*p) return false;
    for (; *p; ++p) {
        if (*p < '0' || *p > '9') return false;
        const unsigned digit = unsigned(*p - '0');
        if (n > (UINT64_MAX - digit) / 10) return false;
        n = n * 10 + digit;
    }
    return true;
}
bool device(char * text, uint64_t & major, uint64_t & minor) noexcept {
    char * colon = strchr(text, ':');
    if (!colon || strchr(colon + 1, ':')) return false;
    *colon = 0;
    return number(text, major) && number(colon + 1, minor);
}
bool token(const char * list, const char * wanted) noexcept {
    const size_t len = strlen(wanted);
    while (*list) {
        const char * end = strchr(list, ',');
        const size_t n = end ? size_t(end-list) : strlen(list);
        if (n == len && !memcmp(list, wanted, n)) return true;
        if (!end) break;
        list = end + 1;
    }
    return false;
}
bool path(const char * p) noexcept {
    if (*p != '/') return false;
    const size_t len = strlen(p);
    if (len == 1) return true;
    if (p[len-1] == '/' || strstr(p, "//")) return false;
    ++p;
    while (*p) {
        const char * e = strchr(p, '/');
        const size_t n = e ? size_t(e-p) : strlen(p);
        if ((n == 1 && *p == '.') || (n == 2 && p[0] == '.' && p[1] == '.')) return false;
        if (!e) break;
        p = e+1;
    }
    return true;
}
bool decode(const char * src, char * dst) noexcept {
    size_t n = 0;
    while (*src) {
        if (n+1 >= path_capacity) return false;
        if (*src == '\\') {
            if (strlen(src) < 4) return false;
            unsigned c = 0;
            for (int i=1; i<=3; ++i) {
                if (src[i] < '0' || src[i] > '7') return false;
                c = c*8 + unsigned(src[i]-'0');
            }
            if (c != 040 && c != 011 && c != 012 && c != 0134) return false;
            dst[n++] = char(c);
            src += 4;
        } else {
            dst[n++] = *src++;
        }
    }
    dst[n] = 0;
    return path(dst);
}
bool within(const char * member, const char * root) noexcept {
    const size_t n = strlen(root);
    return !strcmp(root, "/") || (!strncmp(member, root, n) && (member[n] == 0 || member[n] == '/'));
}
struct lines {
    const char * p;
    size_t left;
    size_t count = 0;
    error next(char * buffer, bool & done) noexcept {
        done = left == 0;
        if (done) return error::ok;
        if (++count > entry_capacity) return error::capacity;
        const char * e = static_cast<const char *>(memchr(p, '\n', left));
        if (!e) return error::malformed;
        const size_t n = size_t(e-p);
        if (n >= line_capacity) return error::capacity;
        memcpy(buffer, p, n);
        buffer[n] = 0;
        p = e+1;
        left -= n+1;
        return error::ok;
    }
};
}
error resolve(const char * mounts, size_t mn, const char * groups, size_t gn, result & out) noexcept {
    out = {};
    out.status = error::malformed;
    if (!mounts || !groups || !mn || !gn) return out.status;
    if (mn > input_capacity || gn > input_capacity) return out.status = error::capacity;
    if (memchr(mounts, 0, mn) || memchr(groups, 0, gn)) return out.status;
    char v1[path_capacity] = {}, v2[path_capacity] = {}, line[line_capacity];
    lines g{groups, gn};
    bool done = false;
    while (true) {
        const auto e = g.next(line, done);
        if (e != error::ok) return out.status = e;
        if (done) break;
        char * a = strchr(line, ':');
        if (!a) return out.status;
        *a++ = 0;
        char * b = strchr(a, ':');
        if (!b) return out.status;
        *b++ = 0;
        uint64_t hierarchy;
        if (!number(line, hierarchy) || strlen(b) >= path_capacity || !path(b)) return out.status;
        char * dst = nullptr;
        if (!*a) {
            if (hierarchy != 0) return out.status;
            dst = v2;
        } else if (token(a, "memory")) {
            if (hierarchy == 0) return out.status;
            dst = v1;
        }
        if (dst) {
            if (*dst) return out.status = error::ambiguous;
            strcpy(dst, b);
        }
    }
    const unsigned version = *v1 ? 1 : (*v2 ? 2 : 0);
    const char * member = version == 1 ? v1 : v2;
    if (!version) return out.status = error::missing;
    lines m{mounts, mn};
    uint64_t seen_ids[entry_capacity];
    size_t n_ids = 0;
    bool found = false, candidate = false, tied = false, have_device = false;
    uint64_t controller_major = 0, controller_minor = 0;
    while (true) {
        const auto e = m.next(line, done);
        if (e != error::ok) return out.status = e;
        if (done) break;
        char * fields[64];
        size_t count = 0;
        char * p = line;
        while (*p) {
            if (count == 64) return out.status = error::capacity;
            fields[count++] = p;
            char * space = strchr(p, ' ');
            if (!space) break;
            *space = 0;
            p = space+1;
            if (!*p || *p == ' ') return out.status;
        }
        size_t sep = 0;
        while (sep < count && strcmp(fields[sep], "-")) ++sep;
        if (sep < 6 || sep+4 != count) return out.status;
        uint64_t id, parent, major, minor;
        if (!number(fields[0], id) || !id || !number(fields[1], parent) || !parent ||
            !device(fields[2], major, minor)) return out.status;
        for (size_t i=0; i<n_ids; ++i) {
            if (seen_ids[i] == id) return out.status = error::ambiguous;
        }
        seen_ids[n_ids++] = id;

        const bool match = version == 2 ? !strcmp(fields[sep+1], "cgroup2") :
            (!strcmp(fields[sep+1], "cgroup") && token(fields[sep+3], "memory"));
        if (!match) continue;
        candidate = true;
        if (have_device && (major != controller_major || minor != controller_minor)) return out.status = error::ambiguous;
        have_device = true;
        controller_major = major;
        controller_minor = minor;
        char root[path_capacity], point[path_capacity];
        if (!strcmp(fields[3], "/..") || !strncmp(fields[3], "/../", 4)) return out.status = error::namespace_unproven;
        if (!decode(fields[3], root) || !decode(fields[4], point)) return out.status;
        if (!within(member, root)) continue;

        if (found) {
            if (!strcmp(root, out.mount_root)) { tied = true; continue; }
            if (within(root, out.mount_root)) continue;
            if (!within(out.mount_root, root)) return out.status = error::ambiguous;
        }
        found = true;
        tied = false;
        out.mount_id = id;
        out.parent_id = parent;
        out.device_major = major;
        out.device_minor = minor;
        out.version = version;
        strcpy(out.mount_root, root);
        strcpy(out.mountpoint, point);
        strcpy(out.member, member);
    }
    if (!found) return out.status = candidate ? error::membership_outside_visible_root : error::missing;
    if (tied) return out.status = error::ambiguous;
    const char * relative = out.member + (!strcmp(out.mount_root, "/") ? 1 : strlen(out.mount_root));
    if (*relative == '/') ++relative;
    strcpy(out.relative_leaf, relative);
    out.visible_root = !*relative;
    out.global_ancestry_proven = false;
    out.widest_visible_mapping = true;
    return out.status = error::ok;
}
}

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace speculative_state {
struct bytes { const uint8_t * data; size_t size; };
inline bool sized(size_t header, size_t count, size_t stride, size_t & total) {
    if (stride && count > (std::numeric_limits<size_t>::max() - header)/stride) { return false; }
    total = header + count*stride;
    return true;
}
struct cursor {
    bytes source;
    size_t offset = 0;
    bool take(size_t n, bytes & out) {
        if (offset > source.size || n > source.size - offset || (n && !source.data)) { return false; }
        out = { source.data ? source.data + offset : nullptr, n };
        offset += n;
        return true;
    }
    bool u32(uint32_t & out) {
        bytes b;
        if (!take(4, b)) { return false; }
        out = uint32_t(b.data[0]) | uint32_t(b.data[1]) << 8 |
              uint32_t(b.data[2]) << 16 | uint32_t(b.data[3]) << 24;
        return true;
    }
    bool end() const { return offset == source.size; }
};
struct envelope { uint8_t owner; bytes payload; };

inline bool walk_envelope(bytes source, envelope & out) {
    cursor c{source}; bytes h, payload;
    if (!c.take(8, h) || h.data[0] != 'S' || h.data[1] != 'P' || h.data[2] != 'C' ||
        h.data[3] != 1 || h.data[5] || h.data[6] || h.data[7] ||
        !c.take(source.size - c.offset, payload) || !c.end()) { return false; }
    out = {h.data[4], payload};
    return true;
}
struct mtp_schema { uint32_t width, mode, heads, window; };
struct mtp_view {
    uint32_t draft_first, target_first, present, position;
    bytes row;
};
inline bool walk_mtp(bytes source, const mtp_schema & schema, mtp_view & out) {
    cursor c{source};
    uint32_t magic, version, width, mode, heads, dfirst, tfirst, window, present, pos;
    if (!c.u32(magic) || !c.u32(version) || !c.u32(width) || !c.u32(mode) ||
        !c.u32(heads) || !c.u32(dfirst) || !c.u32(tfirst) || !c.u32(window) ||
        !c.u32(present) || !c.u32(pos)) { return false; }
    if (magic != 0x3150544d || version != 3 || !width || !heads ||
        width != schema.width || mode != schema.mode || heads != schema.heads || window != schema.window ||
        mode > 7 || (mode & 3) == 3 || dfirst > INT32_MAX || tfirst > INT32_MAX ||
        present > 1 || (present ? pos >= uint32_t(INT32_MAX) : pos != UINT32_MAX)) { return false; }
    size_t total;
    if (!sized(40, present ? size_t(width) : 0, sizeof(float), total) || source.size != total) { return false; }
    bytes row;
    if (!c.take(total - 40, row) || !c.end()) { return false; }
    out = {dfirst, tfirst, present, pos, row};
    return true;
}
struct eagle_view { int32_t position; bytes row; };

inline bool walk_eagle(bytes source, size_t width, eagle_view & out) {
    size_t total;
    if (!width || !sized(sizeof(int32_t), width, sizeof(float), total) || total != source.size) { return false; }
    cursor c{source}; bytes p, row;
    if (!c.take(sizeof(int32_t), p) || !c.take(total - sizeof(int32_t), row) || !c.end()) { return false; }
    int32_t pos;
    std::memcpy(&pos, p.data, sizeof(pos));
    if (pos < -1) { return false; }
    out = {pos, row};
    return true;
}
}

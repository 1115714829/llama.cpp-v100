#pragma once
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

inline float llama_layer_split_fraction(int layer, int first, int count, int repeated) {
    static const bool symmetric = [] {
        const char * value = std::getenv("LLAMA_SYMMETRIC_LAYER_SPLIT");
        return !value || std::strcmp(value, "0") != 0;
    }();
    if (!symmetric) return float(layer - first) / count;
    if (layer >= repeated) return std::nextafter(1.0f, 0.0f);
    return float(layer - first) / std::max(1, std::min(count, repeated - first));
}

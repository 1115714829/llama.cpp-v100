#pragma once

#include <cstdint>

#if defined(GGML_SHARED)
# if defined(_WIN32) && !defined(__MINGW32__)
#  if defined(GGML_BUILD)
#   define GGML_PIN_SETTLE_MEMBER __declspec(dllexport)
#  else
#   define GGML_PIN_SETTLE_MEMBER __declspec(dllimport)
#  endif
# else
#  define GGML_PIN_SETTLE_MEMBER __attribute__((visibility("default")))
# endif
#else
# define GGML_PIN_SETTLE_MEMBER
#endif

namespace ggml_pin_settle {

struct ops {

    int (*settle)(uint64_t token, uint64_t bytes) noexcept;

    void (*release)(uint64_t token, uint64_t bytes) noexcept;
};

GGML_PIN_SETTLE_MEMBER bool register_ops(const ops * table) noexcept;

GGML_PIN_SETTLE_MEMBER const ops * table() noexcept;

}

#undef GGML_PIN_SETTLE_MEMBER

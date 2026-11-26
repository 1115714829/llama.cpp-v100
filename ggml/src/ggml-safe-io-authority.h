#pragma once

#include <cstdint>

#if defined(GGML_SHARED)
# if defined(_WIN32) && !defined(__MINGW32__)
#  if defined(GGML_BUILD)
#   define GGML_SAFE_IO_AUTHORITY_MEMBER __declspec(dllexport)
#  else
#   define GGML_SAFE_IO_AUTHORITY_MEMBER __declspec(dllimport)
#  endif
# else
#  define GGML_SAFE_IO_AUTHORITY_MEMBER __attribute__((visibility("default")))
# endif
#else
# define GGML_SAFE_IO_AUTHORITY_MEMBER
#endif

namespace ggml_safe_io {
namespace authority {

enum outcome : uint8_t {
    outcome_published = 0,
    outcome_cancelled = 1,
    outcome_failed    = 2,
    outcome_external  = 3,

    outcome_read      = 4,
};

struct ops {

    uint64_t (*acquire)(uint64_t bytes, uint64_t * usable) noexcept;

    bool (*release)(uint64_t token, uint64_t bytes, int outcome) noexcept;

    enum class batch_use : uint8_t { pack_pin, pack_write, pack_manifest, manifest_read };
    bool (*acquire_batch)(const uint64_t * bytes, const batch_use * uses, unsigned count,
                          uint64_t * tokens) noexcept = nullptr;
    bool (*claim_batch)(uint64_t token, uint64_t bytes, batch_use use) noexcept = nullptr;
    void (*cancel_unclaimed)(uint64_t token, uint64_t bytes) noexcept = nullptr;
};

GGML_SAFE_IO_AUTHORITY_MEMBER bool authority_register(const ops * table) noexcept;

GGML_SAFE_IO_AUTHORITY_MEMBER const ops * authority_table() noexcept;

}
}

#undef GGML_SAFE_IO_AUTHORITY_MEMBER

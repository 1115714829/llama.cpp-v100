#pragma once

#include <cstddef>
#include <cstdint>

namespace ggml_expert_pack {

enum class phase : uint8_t {
    idle      = 0,
    writing   = 1,
    published = 2,
    reading   = 3,
};

class pack_file {
public:

    bool begin_write() noexcept {
        if (pinned_ || addr_ != nullptr) { return false; }
        phase_ = phase::writing;
        return true;
    }

    bool begin_read() noexcept {
        if (pinned_) { return false; }
        phase_ = phase::reading;
        return true;
    }

    void publish() noexcept { phase_ = phase::published; }

    void mapped(void * addr, size_t bytes) noexcept { addr_ = addr; bytes_ = bytes; }
    void * addr() const noexcept { return addr_; }
    size_t bytes() const noexcept { return bytes_; }
    bool is_mapped() const noexcept { return addr_ != nullptr; }

    bool may_pin() const noexcept {
        return phase_ == phase::published || phase_ == phase::reading;
    }

    bool may_write() const noexcept {
        return phase_ == phase::writing && !pinned_;
    }
    bool is_pinned() const noexcept { return pinned_; }
    phase get() const noexcept { return phase_; }

    template <typename Op>
    bool pin(Op op) noexcept {
        if (!may_pin() || pinned_) { return false; }
        if (!op()) { return false; }
        pinned_ = true;
        return true;
    }

    template <typename Unpin, typename Unmap>
    bool release(Unpin unpin, Unmap unmap) noexcept {
        if (addr_ == nullptr) {
            phase_ = phase::idle;
            return true;
        }
        if (pinned_) {
            if (!unpin(addr_)) { return false; }
            pinned_ = false;
        }
        if (!unmap(addr_, bytes_)) { return false; }
        addr_ = nullptr;
        bytes_ = 0;
        phase_ = phase::idle;
        return true;
    }

private:
    phase  phase_  = phase::idle;
    bool   pinned_ = false;
    void * addr_   = nullptr;
    size_t bytes_  = 0;
};

}

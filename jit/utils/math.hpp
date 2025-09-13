#pragma once

#include <torch/python.h>

#include "exception.hpp"

namespace kernels {

template <typename T> static constexpr T ceil_div(const T& a, const T& b) {
    return (a + b - 1) / b;
}

template <typename T> static constexpr T align(const T& a, const T& b) {
    return ceil_div(a, b) * b;
}

inline int get_tma_aligned_size(const int x, const int element_size) {
    constexpr int kNumTMAAlignmentBytes = 16;
    K_HOST_ASSERT(kNumTMAAlignmentBytes % element_size == 0);
    return align(x, kNumTMAAlignmentBytes / element_size);
}

} // namespace kernels

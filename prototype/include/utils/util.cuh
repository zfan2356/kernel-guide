#pragma once

namespace kernels::prototype::utils {

template <typename T, typename... args>
__device__ __forceinline__ static size_t size_bytes(const T& _1, const args&... _2) {
    return sizeof(T) + size_bytes(_2...);
}

__host__ __device__ __forceinline__ static constexpr uint32_t ceil_div(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

} // namespace kernels::prototype::utils

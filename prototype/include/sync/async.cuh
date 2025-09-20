#pragma once

#include <cuda_runtime.h>

namespace kernels::prototype::async {
// Concept to constrain Bytes parameter for cp.async
// cp-size can only be 4, 8 and 16 bytes
template <uint32_t Bytes> concept ValidCpAsyncBytes = (Bytes == 4 || Bytes == 8 || Bytes == 16);

struct CpAsync {
    enum class CacheOperator {
        OpCA,
        OpCG,
    };
    template <uint32_t Bytes, CacheOperator Op>
    requires ValidCpAsyncBytes<Bytes> __device__ __forceinline__ static void call(void* dst, const void* src) {
        uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
        if constexpr (Op == CacheOperator::OpCG) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst_ptr), "l"(src), "n"(Bytes) : "memory");
        } else {
            asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(dst_ptr), "l"(src), "n"(Bytes) : "memory");
        }
    }

    __device__ __forceinline__ static void commit_group() {
        asm volatile("cp.async.commit_group ;");
    }

    template <uint32_t N = 0> __device__ __forceinline__ static void wait_group() {
        if constexpr (N == 0) {
            asm volatile("cp.async.wait_all ;");
        } else {
            asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
        }
    }
};

} // namespace kernels::prototype::async

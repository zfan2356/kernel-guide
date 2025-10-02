#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include "../runtime/runtime.cuh"
#include <cuda_bf16.h>
#include <cassert>

#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/arch/mma_sm90_gmma_ext.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

namespace kernels::prototype::async {
// Constrain Bytes parameter for cp.async: allowed sizes are 4, 8, and 16 bytes
template <uint32_t Bytes> concept ValidCpAsyncBytes = (Bytes == 4 || Bytes == 8 || Bytes == 16);

struct CpAsync {
    enum class CacheOperator {
        OpCA,
        OpCG,
    };
    template <uint32_t Bytes, CacheOperator Op>
    requires ValidCpAsyncBytes<Bytes> __device__ __forceinline__ static void call(
        void* dst, const void* src) {
        uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
        if constexpr (Op == CacheOperator::OpCG) {
            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst_ptr), "l"(src), "n"(Bytes)
                : "memory");
        } else {
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(dst_ptr), "l"(src), "n"(Bytes)
                : "memory");
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


/*
    TMA (Tensor Memory Accelerator) requires an SM90 or newer compute architecture.
    It utilizes mbarrier synchronization to coordinate threads.
    The implementation here is a group-level TMA, where only one thread in the group
    is responsible for committing the group operation.
    Since TMA is a specialized hardware component, it is sufficient for a single thread to trigger
   the operation, similar to how a DMA (Direct Memory Access) controller works. TMA has two types of
   PTX
    - cp.async.bulk: to async load/store 1d data betwen global and shared memory
    - cp.async.bulk.tensor: to async load/store 3d/4d/5d data betwen global and shared memory
                         need tensor map to describe the shape and layout of the tensor
*/
struct TMA {
    __device__ __forceinline__ static void load_1d(
        void* dst, const void* src, uint32_t nBytes, uint64_t* bar_ptr) {
        // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk
        assert(nBytes % 16 == 0);
        uint64_t src_ptr = reinterpret_cast<uint64_t>(src);
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                     "[%0], [%1], %2, [%3];\n"
            :
            : "l"(__cvta_generic_to_shared(dst)), "l"(src_ptr), "r"(nBytes),
            "l"(__cvta_generic_to_shared(bar_ptr)));
    }

    __device__ __forceinline__ static void prefetch_tensor_map(const void* desc) {
        uint64_t gmem_desc = reinterpret_cast<uint64_t>(desc);
        asm volatile("prefetch.tensormap [%0];" : : "l"(gmem_desc) : "memory");
    }

    __device__ __forceinline__ static void load_2d(
        void* dst, const void* desc, uint64_t* mbar, uint2 coord) {
        uint64_t gmem_desc = reinterpret_cast<uint64_t>(desc);
        uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%3, %4}], [%2];" ::"r"(dst_ptr),
            "l"(gmem_desc), "l"(__cvta_generic_to_shared(mbar)), "r"(coord.y), "r"(coord.x)
            : "memory");
    }

    __device__ __forceinline__ static void store_1d(void* dst, const void* src, uint32_t nBytes) {
        // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk
        // it seems that cp.async.bulk.global.shared::cta not support mbarrier...
        // it only support commit group and wait group
        assert(nBytes % 16 == 0);
        uint64_t dst_ptr = reinterpret_cast<uint64_t>(dst);
        asm volatile("cp.async.bulk.global.shared::cta.bulk_group "
                     "[%0], [%1], %2;\n"
            :
            : "l"(dst_ptr), "l"(__cvta_generic_to_shared(src)), "r"(nBytes));
    }

    __device__ __forceinline__ static void store_2d(const void* desc, void* src, uint2 coord) {
        uint64_t gmem_desc = reinterpret_cast<uint64_t>(desc);
        uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(src));
        asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
            :
            : "l"(gmem_desc), "r"(src_ptr), "r"(coord.y), "r"(coord.x)
            : "memory");
    }

    __device__ __forceinline__ static void commit_group() {
        asm volatile("cp.async.bulk.commit_group;");
    }

    template <int N = 0> __device__ __forceinline__ static void store_async_wait() {
        asm volatile("cp.async.bulk.wait_group %0;" : : "n"(N) : "memory");
    }
};

} // namespace kernels::prototype::async

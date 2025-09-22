#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include "../runtime/runtime.cuh"
#include <cuda_bf16.h>

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


/*
    TMA (Tensor Memory Accelerator) requires an SM90 or newer compute architecture.
    It utilizes mbarrier synchronization to coordinate threads.
    The implementation here is a group-level TMA, where only one thread in the group
    is responsible for committing the group operation.
    Since TMA is a specialized hardware component, it is sufficient for a single thread to trigger the operation,
    similar to how a DMA (Direct Memory Access) controller works.
    TMA has two types of PTX
    - cp.async.bulk: to async load/store 1d data betwen global and shared memory
    - cp.async.bulk.tensor: to async load/store 3d/4d/5d data betwen global and shared memory
                         need tensor map to describe the shape and layout of the tensor
*/
struct TMA {
    /*
        create a tensor map for the given source tensor
        tensor map is a descriptor of the tensor in memory
        it contains the shape and layout of the tensor
    */
    template <typename T> struct TensorMap {
        constexpr static int tensor_dim = 5;
        __host__ __forceinline__ static CUtensorMap* create(T* src, int batch, int depth, int rows, int cols) {
            void* global_addr = (void*)(src);

            // now we only support bfloat16 and float
            constexpr CUtensorMapDataType tma_format =
                (std::is_same_v<T, __nv_bfloat16> ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
                 : std::is_same_v<T, float>       ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32
                                                  : CUtensorMapDataType(-1));

            constexpr CUtensorMapInterleave tma_interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
            constexpr CUtensorMapL2promotion tma_l2promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
            constexpr CUtensorMapFloatOOBfill tma_oobfill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
            //
            constexpr CUtensorMapSwizzle tma_swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;

            uint32_t gmem_shape[tensor_dim] = {0, 0, 0, 0, 0};
            uint32_t gmem_stride[tensor_dim - 1] = {0, 0, 0, 0};
            uint32_t smem_shape[tensor_dim] = {0, 0, 0, 0, 0};
            uint32_t smem_stride[tensor_dim] = {1, 1, 1, 1, 1};

            // constexpr uint32_t shae
        }
    };

    template <uint32_t Dim>
    __device__ __forceinline__ static void load(void* dst, const void* src, uint32_t nbBytes, void* bar) {
        if constexpr (Dim == 1) {
            if (runtime::elect_one_sync()) {
                asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                             "[%0], [%1], %2, [%3];\n"
                             :
                             : "l"(__cvta_generic_to_shared(dst)), "l"(reinterpret_cast<uint64_t>(src)), "r"(nbBytes),
                               "l"(__cvta_generic_to_shared(&bar)));
            }
        } else {
            static_assert(false, "Only support 1D tensor map for now");
        }
    }

    __device__ __forceinline__ static void commit_group() {
        if (runtime::elect_one_sync()) {
            asm volatile("cp.async.bulk.commit_group;");
        }
        __syncwarp();
    }

    template <int N = 0> __device__ __forceinline__ static void store_async_wait() {
        if (runtime::elect_one_sync()) {
            asm volatile("cp.async.bulk.wait_group %0;" : : "n"(N) : "memory");
        }
        __syncwarp();
    }

    template <int N = 0> __device__ __forceinline__ static void read_async_wait() {
        if (runtime::elect_one_sync()) {
            asm volatile("cp.async.bulk.wait_group.read %0;" : : "n"(N) : "memory");
        }
        __syncwarp();
    }
};

} // namespace kernels::prototype::async

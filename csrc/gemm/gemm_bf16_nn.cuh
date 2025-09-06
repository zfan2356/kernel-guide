#include <assert.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iostream>

namespace gemm {
using bf16 = __nv_bfloat16;
int _PRE_M = 100;
namespace utils {
    __device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) {
        return (((x) & 0x3FFFF) >> 0x4);
    }

    __device__ uint64_t make_smem_desc(bf16* ptr) {
        uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
        uint64_t desc = 0x0000000000000000;
        desc |= matrix_descriptor_encode(addr);
        desc |= matrix_descriptor_encode((uint64_t)16) << 16;
        desc |= matrix_descriptor_encode((uint64_t)1024) << 32;
        desc |= 1llu << 62;
        return desc;
    }

    __device__ void warpgroup_arrive() {
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    }

    __device__ void warpgroup_commit_batch() {
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    }

    template <int N> __device__ void warpgroup_wait() {
        static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be range [0, 7]");
        asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
    }

    template <uint32_t BlockMajorSize, uint32_t BlockMinorSize, bool swizzle = true>
    __host__ static inline CUtensorMap create_tensor_map(bf16* ptr, int global_height, int global_width) {
        CUtensorMap tma_map;
        void* gmem_addr = (void*)ptr;
        static_assert(BlockMinorSize >= 64);
        assert(global_width % 64 == 0);

        uint64_t gmem_prob_shape[5] = {64, (uint64_t)global_height, (uint64_t)global_width / 64, 1, 1};
        uint64_t gmem_prob_stride[5] = {sizeof(bf16) * global_width, 64 * sizeof(bf16), 0, 0, 0};
        uint32_t smem_box_shape[5] = {64, BlockMajorSize, BlockMinorSize / 64, 1, 1};
        uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

        CUresult res =
            cuTensorMapEncodeTiled(&tma_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, gmem_addr, gmem_prob_shape,
                                   gmem_prob_stride, smem_box_shape, smem_box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                   swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
                                   CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        assert(res == CUDA_SUCCESS);
        return tma_map;
    }

} // namespace utils

template <uint32_t N, uint32_t K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t SUPER_N,
          uint32_t NUM_M_BLOCKS, uint32_t CLUSTER_M, uint32_t CLUSTER_N, uint32_t NUM_STAGES>
struct GeMMBF16NN {
    static void Run(bf16* out, bf16* a, bf16* b, uint32_t m, cudaStream_t stream, int NUM_SMS) {
        static_assert(NUM_SMS % (CLUSTER_M * CLUSTER_N) == 0, "num_sms must be divisible by CLUSTER_M * CLUSTER_N");

        if (_PRE_M != m) {
            _PRE_M = m;
            d_tma_map_b = create_tensor_map<BLOCK_M, BLOCK_K>(a, m, K);
            d_tma_map_a = create_tensor_map<BLOCK_K, BLOCK_N>(b, K, N);
            d_tma_map_out = create_tensor_map<BLOCK_M, BLOCK_N>(out, m, N);
        }
    }
};
} // namespace gemm

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_bf16.h>
#include "prototype.cuh"
#include <cutlass/arch/barrier.h>

namespace kernels::tma {
using namespace kernels::prototype;
using bf16_2 = nv_bfloat162;
using bf16 = nv_bfloat16;

using Barrier = sync::Semaphore;
// Alternative: used for debugging
// using Barrier = cutlass::arch::ClusterTransactionBarrier;

template <typename Derived> struct TmaStrategy {
    __device__ __forceinline__ static void tma_async_load(
        bf16* smem_x, bf16* x, uint32_t* barrier_ptr, uint2 coord, const void* tensor_map) {
        // Load a BlockM x BlockN tile into shared memory via TMA.
        // A single thread triggers the TMA instruction, which executes on dedicated hardware.
        // Note: `cp.async.bulk` does not require a tensor map descriptor, whereas
        // `cp.async.bulk.tensor.2d` does.

        Derived::tma_async_load(smem_x, x, barrier_ptr, coord, tensor_map);
    }

    __device__ __forceinline__ static void compute_and_store(
        bf16* smem_x, bf16* smem_y, bf16* out, uint2 coord, const void* tensor_map) {
        // Perform computation on data resident in shared memory.
        //
        // Each thread processes 2 bf16 elements per register lane (NumElem bf16_2 pairs).
        // The kernel computes y = 2 * x + 1 in registers, stores results back to shared memory,
        // then uses TMA to write the final output to global memory.
        //
        // Note: Writing from registers directly to global memory would be faster, but routing
        // through shared memory is used here to demonstrate TMA store functionality.
        Derived::compute_and_store(smem_x, smem_y, out, coord, tensor_map);
    }
};

// NumElem: number of bf16_2 elements processed per thread in registers
// BlockM/BlockN: tile shape
// M/N: problem shape
template <uint32_t kNumElem, uint32_t BlockM, uint32_t BlockN, uint32_t M, uint32_t N, uint32_t SwizzleXMode, uint32_t SwizzleOutMode>
struct TmaStrategyV1 : public TmaStrategy<TmaStrategyV1<kNumElem, BlockM, BlockN, M, N, SwizzleXMode, SwizzleOutMode>> {
    struct Register {
        bf16_2 regs[kNumElem];
        __device__ __forceinline__ void load(void* src, uint32_t offs) {
            using copy = memory::Move<bf16_2, memory::CopyAtom::OpS2R>;
            void* src_ptr = reinterpret_cast<void*>(reinterpret_cast<bf16*>(src) + offs);
            copy::move<kNumElem>(regs, src_ptr);
        }
        __device__ __forceinline__ void compute() {
            auto mul2_add1 = [&](bf16_2& v) {
                v = __hadd2(__hmul2(v, bf16_2(2, 2)), bf16_2(1, 1));
            };
            #pragma unroll
            for (uint32_t i = 0; i < kNumElem; i++) {
                mul2_add1(regs[i]);
            }
        }
        __device__ __forceinline__ void store(void* dst, uint32_t offs) {
            using copy = memory::Move<bf16_2, memory::CopyAtom::OpR2S>;
            void* dst_ptr = reinterpret_cast<void*>(reinterpret_cast<bf16*>(dst) + offs);
            copy::move<kNumElem>(dst_ptr, regs);
        }
    };

    __device__ __forceinline__ static void tma_async_load(bf16* smem_x, bf16* x,
        uint32_t* barrier_ptr, uint32_t crd0, uint32_t crd1, const void* tensor_map) {
        auto global_offset = (crd0 * BlockM) * N + crd1 * BlockN;
        for (uint32_t i = 0; i < BlockM; i++) {
            bf16* smem_ptr = smem_x + i * BlockN;
            async::TMA::load_1d(
                smem_ptr, &x[global_offset + i * N],
                BlockN * sizeof(bf16), barrier_ptr
            );
        }
    }

    __device__ __forceinline__ static void compute_and_store(bf16* smem_x, bf16* smem_y, bf16* out,
        uint32_t crd0, uint32_t crd1, const void* tensor_map) {
        Register regs{};
        // Perform computation within a single warp
        static constexpr auto kBlockSize = BlockM * BlockN;
        static constexpr auto kWarpStep = kNumElem * 32 * 2;
        #pragma unroll
        for (int i = runtime::laneid() * kNumElem * 2; i < kBlockSize; i += kWarpStep) {
            regs.load(smem_x, i);
            regs.compute();
            regs.store(smem_y, i);
        }
        async::TMA::tma_store_fence();

        if (runtime::elect_one_sync()) {
            auto global_offset = (crd0 * BlockM) * N + crd1 * BlockN;
            #pragma unroll
            for (uint32_t i = 0; i < BlockM; i++) {
                bf16* smem_y_ptr = smem_y + i * BlockN;
                async::TMA::store_1d(
                    &out[global_offset + i * N], smem_y_ptr, BlockN * sizeof(bf16)
                );
            }
            async::TMA::tma_commit_group();
            async::TMA::tma_store_wait<0>();
        }
    }
};

// NumElem: number of bf16_2 elements processed per thread in registers
// BlockM/BlockN: tile dimensions
// M/N: global problem dimensions
// Uses cp.async.bulk.tensor.2d for 2D asynchronous load/store operations
template <uint32_t kNumElem, uint32_t BlockM, uint32_t BlockN, uint32_t M, uint32_t N, uint32_t SwizzleXMode, uint32_t SwizzleOutMode>
struct TmaStrategyV2 : public TmaStrategy<TmaStrategyV2<kNumElem, BlockM, BlockN, M, N, SwizzleXMode, SwizzleOutMode>> {
    using CopyAtom = memory::CopyAtom;
    template <CopyAtom LoadCopyAtom = CopyAtom::OpS2R, CopyAtom StoreCopyAtom = CopyAtom::OpR2S>
    struct Register {
        bf16_2 regs[kNumElem];
        __device__ __forceinline__ void load(void* src, uint32_t offs) {
            using copy = memory::Move<bf16_2, LoadCopyAtom>;
            void* src_ptr = reinterpret_cast<void*>(reinterpret_cast<bf16*>(src) + offs);
            copy::move<kNumElem>(regs, src_ptr);
        }
        __device__ __forceinline__ void compute() {
            auto mul2_add1 = [&](bf16_2& v) {
                v = __hadd2(__hmul2(v, bf16_2(2, 2)), bf16_2(1, 1));
            };
            #pragma unroll
            for (uint32_t i = 0; i < kNumElem; i++) {
                mul2_add1(regs[i]);
            }
        }
        __device__ __forceinline__ void store(void* dst, uint32_t offs) {
            using copy = memory::Move<bf16_2, StoreCopyAtom>;
            void* dst_ptr = reinterpret_cast<void*>(reinterpret_cast<bf16*>(dst) + offs);
            copy::move<kNumElem>(dst_ptr, regs);
        }
    };

    __device__ __forceinline__ static void tma_async_load(bf16* smem_x, bf16* x,
        uint32_t* barrier_ptr, uint32_t crd0, uint32_t crd1, const void* tensor_map) {
        crd0 *= BlockN, crd1 *= BlockM;
        // constexpr auto cache_hint = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);
        // cute::SM90_TMA_LOAD_2D::copy(tensor_map, reinterpret_cast<uint64_t*>(barrier_ptr),
        //                                 cache_hint, smem_x, crd0, crd1);
        async::TMA::load_2d(smem_x, tensor_map, barrier_ptr, crd0, crd1);
    }

    __device__ __forceinline__ static void compute_and_store(bf16* smem_x, bf16* smem_y, bf16* out,
        uint32_t crd0, uint32_t crd1, const void* tensor_map) {
        if (runtime::elect_one_sync()) {
            // Wait for previous store instructions to complete
            async::TMA::tma_store_wait<0>();
        }
        crd0 *= BlockN, crd1 *= BlockM;
        Register regs{};
        static constexpr auto kBlockSize = BlockM * BlockN;
        static constexpr auto kWarpStep = kNumElem * 32 * 2;

        // calc swizzle mode
        if constexpr (SwizzleXMode != SwizzleOutMode) {
            // x to non-swizzle layout
        }
        #pragma unroll
        for (int i = runtime::laneid() * kNumElem * 2; i < kBlockSize; i += kWarpStep) {
            regs.load(smem_x, i);
            regs.compute();
            regs.store(smem_y, i);
        }
        async::TMA::tma_store_fence();

        if (runtime::elect_one_sync()) {
            async::TMA::store_2d(tensor_map, smem_y, crd0, crd1);
            async::TMA::tma_commit_group();
        }
    }
};

template<uint32_t SwizzleMode>
struct SwizzlePattern {
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#the-swizzle-modes
    // This Functor will implement the swizzle pattern for the given SwizzleMode
    static constexpr uint32_t kBytes = sizeof(bf16);
    static constexpr uint32_t kBankGroupBytes = 16; // it fixed to 16 bytes for document
    static constexpr uint32_t kBankBytes = 4; // in shared memory, each bank is 4 bytes
    static constexpr uint32_t kInnerDimBlock = SwizzleMode / kBytes;
    __device__ __forceinline__ int2 operator()(int i, int j) const {
        // particularly for x, we have [BlockM, BlockN] shape for non-swizzle layout,
        // but for swizzle layout, we have [BlockM, kInnerDimBlock] shape, -> [kInnerDimBlock, BlockM]

        auto swizzle_128b = [&](int i, int j) {
            // swizzle 128 bytes means that we will interleave 128 / 16 = 8 bank groups
            // now BlockN is 64 bfloat16 -> 128 bytes
            // so it will wrap 8 bfloat16 to one bank group, and interleave 8 "bank group rows" => 64 rows
            // 128 bytes is suitable for 64 * 64 shared memory block
        };


        switch (SwizzleMode) {
            case 0:
            return {i, j};
            case 16:
            return {i, j};
            case 32:
            return {i, j};
            case 64:
            return {i, j};
            case 128:
            return {i, j};
        }
    }
};

} // namespace kernels::tma

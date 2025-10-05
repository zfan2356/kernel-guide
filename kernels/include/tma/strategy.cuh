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
// using Barrier = cutlass::arch::ClusterTransactionBarrier;

template <typename Derived> struct TmaStrategy {
    __device__ __forceinline__ static void tma_async_load(
        bf16* smem_x, bf16* x, uint32_t* barrier_ptr, uint2 coord, CUtensorMap tensor_map) {
        Derived::tma_async_load(smem_x, x, barrier_ptr, coord, tensor_map);
    }

    __device__ __forceinline__ static void compute_and_store(
        bf16* smem_x, bf16* smem_y, bf16* out, uint2 coord, CUtensorMap tensor_map) {
        Derived::compute_and_store(smem_x, smem_y, out, coord, tensor_map);
    }
};

// NumElem: number of bf16_2 elements processed per thread in registers
// BlockM/BlockN: tile shape
// M/N: problem shape
template <uint32_t NumElem, uint32_t BlockM, uint32_t BlockN, uint32_t M, uint32_t N>
struct TmaStrategyV1 : public TmaStrategy<TmaStrategyV1<NumElem, BlockM, BlockN, M, N>> {
    struct Register {
        bf16_2 regs[NumElem];
        __device__ __forceinline__ void load(void* src, uint32_t offs) {
            using copy = memory::Move<bf16_2, memory::CopyAtom::OpS2R>;
            void* src_ptr = reinterpret_cast<void*>(reinterpret_cast<bf16*>(src) + offs);
            copy::move<NumElem>(regs, src_ptr);
        }
        __device__ __forceinline__ void compute() {
            for (uint32_t i = 0; i < NumElem; i++) {
                regs[i] = __hmul2(regs[i], bf16_2(2, 2));
                regs[i] = __hadd2(regs[i], bf16_2(1, 1));
            }
        }
        __device__ __forceinline__ void store(void* dst, uint32_t offs) {
            using copy = memory::Move<bf16_2, memory::CopyAtom::OpR2S>;
            void* dst_ptr = reinterpret_cast<void*>(reinterpret_cast<bf16*>(dst) + offs);
            copy::move<NumElem>(dst_ptr, regs);
        }
    };

    __device__ __forceinline__ static void tma_async_load(bf16* smem_x, bf16* x,
        uint32_t* barrier_ptr, uint32_t crd0, uint32_t crd1, CUtensorMap tensor_map) {
        auto global_offset = (crd0 * BlockM) * N + crd1 * BlockN;
        for (uint32_t i = 0; i < BlockM; i++) {
            bf16* smem_ptr = smem_x + i * BlockN;
            async::TMA::load_1d(
                smem_ptr, &x[global_offset + i * N], BlockN * sizeof(bf16), barrier_ptr);
        }
    }

    __device__ __forceinline__ static void compute_and_store(bf16* smem_x, bf16* smem_y, bf16* out,
        uint32_t crd0, uint32_t crd1, CUtensorMap tensor_map) {
        Register regs;
        // do calc in one warp
#pragma unroll
        for (int i = runtime::laneid() * NumElem * 2; i < BlockM * BlockN; i += NumElem * 32 * 2) {
            regs.load(smem_x, i);
            regs.compute();
            regs.store(smem_y, i);
        }
        __syncwarp();

        if (runtime::elect_one_sync()) {
            auto global_offset = (crd0 * BlockM) * N + crd1 * BlockN;
#pragma unroll
            for (uint32_t i = 0; i < BlockM; i++) {
                bf16* smem_y_ptr = smem_y + i * BlockN;
                async::TMA::store_1d(
                    &out[global_offset + i * N], smem_y_ptr, BlockN * sizeof(bf16));
            }
            async::TMA::commit_group();
            async::TMA::store_async_wait();
        }
        __syncwarp();
    }
};

// NumElem: number of bf16_2 elements processed per thread in registers
// BlockM/BlockN: tile shape
// M/N: problem shape
// use cp.async.bulk.tensor.2d to implement 2d async load/store
template <uint32_t NumElem, uint32_t BlockM, uint32_t BlockN, uint32_t M, uint32_t N>
struct TmaStrategyV2 : public TmaStrategy<TmaStrategyV2<NumElem, BlockM, BlockN, M, N>> {
    struct Register {
        bf16_2 regs[NumElem];
        __device__ __forceinline__ void load(void* src, uint32_t offs) {
            using copy = memory::Move<bf16_2, memory::CopyAtom::OpS2R>;
            void* src_ptr = reinterpret_cast<void*>(reinterpret_cast<bf16*>(src) + offs);
            copy::move<NumElem>(regs, src_ptr);
        }
        __device__ __forceinline__ void compute() {
            for (uint32_t i = 0; i < NumElem; i++) {
                regs[i] = __hmul2(regs[i], bf16_2(2, 2));
                regs[i] = __hadd2(regs[i], bf16_2(1, 1));
            }
        }
        __device__ __forceinline__ void store(void* dst, uint32_t offs) {
            using copy = memory::Move<bf16_2, memory::CopyAtom::OpR2S>;
            void* dst_ptr = reinterpret_cast<void*>(reinterpret_cast<bf16*>(dst) + offs);
            copy::move<NumElem>(dst_ptr, regs);
        }
    };

    __device__ __forceinline__ static void tma_async_load(bf16* smem_x, bf16* x,
        uint32_t* barrier_ptr, uint32_t crd0, uint32_t crd1, CUtensorMap tensor_map) {
        uint2 global_coord = {crd0 * BlockM, crd1 * BlockN};
        async::TMA::load_2d(smem_x, &tensor_map, barrier_ptr, global_coord);
    }

    __device__ __forceinline__ static void compute_and_store(bf16* smem_x, bf16* smem_y, bf16* out,
        uint32_t crd0, uint32_t crd1, CUtensorMap tensor_map) {
        Register regs;
        uint2 global_coord = {crd0 * BlockM, crd1 * BlockN};
#pragma unroll
        for (int i = runtime::laneid() * NumElem * 2; i < BlockM * BlockN; i += NumElem * 32 * 2) {
            regs.load(smem_x, i);
            regs.compute();
            regs.store(smem_y, i);
        }
        __syncwarp();

        if (runtime::elect_one_sync()) {
            async::TMA::store_2d(&tensor_map, smem_y, global_coord);
            async::TMA::commit_group();
            async::TMA::store_async_wait();
        }
        __syncwarp();
    }
};

} // namespace kernels::tma

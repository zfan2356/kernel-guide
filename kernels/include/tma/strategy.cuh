#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_bf16.h>
#include "prototype.cuh"


namespace kernels::tma {
using namespace kernels::prototype;
using bf16_2 = __nv_bfloat162;
using bf16 = __nv_bfloat16;
using Barrier = sync::Semaphore;

template <typename Derived> struct TmaStrategy {
    __device__ __forceinline__ static void tma_async_load(bf16* smem_x, bf16* x, uint64_t* barrier_ptr, uint2 coord) {
        Derived::tma_async_load(smem_x, x, barrier_ptr, coord);
    }

    __device__ __forceinline__ static void compute_and_store(
        bf16* smem_x, bf16* smem_y, bf16* out, uint32_t stage_id, uint2 coord) {
        Derived::compute_and_store(smem_x, smem_y, out, stage_id, coord);
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
                regs[i] = __hmul2(regs[i], __nv_bfloat162(2, 2));
                regs[i] = __hadd2(regs[i], __nv_bfloat162(1, 1));
            }
        }
        __device__ __forceinline__ void store(void* dst, uint32_t offs) {
            using copy = memory::Move<bf16_2, memory::CopyAtom::OpR2S>;
            void* dst_ptr = reinterpret_cast<void*>(reinterpret_cast<bf16*>(dst) + offs);
            copy::move<NumElem>(dst_ptr, regs);
        }
    };

    __device__ __forceinline__ static void tma_async_load(bf16* smem_x, bf16* x, uint64_t* barrier_ptr, uint2 coord) {
        auto global_offset = (coord.x * BlockM) * N + coord.y * BlockN;
#pragma unroll
        for (uint32_t i = 0; i < BlockM; i++) {
            bf16* smem_ptr = smem_x + i * BlockN;
            async::TMA::load<1>(&smem_ptr[0], &x[global_offset + i * N], BlockN * sizeof(bf16), barrier_ptr);
        }
    }

    __device__ __forceinline__ static void compute_and_store(bf16* smem_x, bf16* smem_y, bf16* out, uint2 coord) {
        Register regs;
#pragma unroll
        for (int i = runtime::laneid() * NumElem * 2; i < BlockM * BlockN; i += NumElem * 32 * 2) {
            regs.load(smem_x, i);
            regs.compute();
            regs.store(smem_y, i);
        }

        if (runtime::elect_one_sync()) {
            auto global_offset = (coord.x * BlockM) * N + coord.y * BlockN;
#pragma unroll
            for (uint32_t i = 0; i < BlockM; i++) {
                bf16* smem_y_ptr = smem_y + i * BlockN;
                async::TMA::store<1>(&out[global_offset + i * N], &smem_y_ptr[0], BlockN * sizeof(bf16));
            }
            async::TMA::commit_group();
            async::TMA::store_async_wait();
        }
    }
};

} // namespace kernels::tma

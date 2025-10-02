#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_bf16.h>
#include "prototype.cuh"

#include <cutlass/arch/barrier.h>

namespace kernels::tma {
using namespace kernels::prototype;
using bf16_2 = __nv_bfloat162;
using bf16 = __nv_bfloat16;
using Barrier = sync::Semaphore;

template <uint32_t NumWorkers, uint32_t M, uint32_t N, uint32_t BlockM, uint32_t BlockN, uint32_t NumBlocks,
          uint32_t NumStages>
struct Scheduler {
    constexpr static auto NumWarps = NumWorkers * NumBlocks;

    explicit __device__ Scheduler(uint32_t initial_rank) {
        num_m_blocks = utils::ceil_div(M, BlockM);
        num_n_blocks = utils::ceil_div(N, BlockN);
        num_blocks = num_m_blocks * num_n_blocks;
        current_block = blockIdx.x * NumWorkers + initial_rank;
        current_iter = 0;
        stage_id = initial_rank;
    }
    __device__ __forceinline__ bool current_work_tile(uint32_t& m_idx, uint32_t& n_idx) const {
        if (current_block >= num_blocks) {
            return false;
        }
        m_idx = current_block / num_n_blocks;
        n_idx = current_block % num_n_blocks;
        return true;
    }
    __device__ __forceinline__ void step() {
        current_block += NumWarps;
        current_iter++;
        stage_id = (stage_id + NumWorkers) % NumStages;
    }

    uint32_t num_blocks, num_m_blocks, num_n_blocks;
    uint32_t current_block;
    int current_iter, stage_id;
};

template <uint32_t NumBlocks, uint32_t NumWarpsPerBlock, uint32_t NumConsumers, uint32_t NumProducers,
          uint32_t NumStages, uint32_t M, uint32_t N, uint32_t BlockM, uint32_t BlockN>
__global__ __launch_bounds__(NumWarpsPerBlock * 32, 1) void tma_impl(bf16* x, bf16* out) {
    /*
    Overview:
      - x: [M, N] bfloat16
      - out: [M, N] bfloat16
      - 1 consumer warp, 1 producer warp
      - NumStages = 4

    This kernel copies tiles of x to shared memory, computes y = 2 * x + 1,
    then writes the results to out. It uses TMA to asynchronously load/store
    between global and shared memory in order to overlap memory transfers with
    computation. Reference:
      https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies-using-the-tensor-memory-accelerator-tma

    Notes:
      - TMA is effectively a superset of cp.async. While cp.async provides
        asynchronous loads only, TMA supports both asynchronous loads and
        stores, enabling fuller overlap of memory and compute.

    Roadmap:
      - v1: use `cp.async.bulk` as a simple drop-in to implement 1D async
        load/store.
      - v2: use `cp.async.bulk.tensor.2d` with a tensor map descriptor to load
        2D tiles asynchronously.

    We adopt a warp-specialized producer–consumer design.
    */
    constexpr static uint32_t NumWarps = NumWarpsPerBlock * NumBlocks;
    static_assert(NumConsumers + NumProducers == NumWarpsPerBlock);

    // Initialize multi-stage shared memory
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    constexpr static auto SMEM_SIZE = BlockM * BlockN * sizeof(bf16);
    static_assert(SMEM_SIZE % 1024 == 0, "SMEM_SIZE must be aligned to 1024");

    bf16* smem_x[NumStages];
    bf16* smem_y[NumStages];
    Barrier* empty_barrier[NumStages];
    Barrier* full_barrier[NumStages];

#pragma unroll
    for (int i = 0; i < NumStages; i++) {
        smem_x[i] = reinterpret_cast<bf16*>(smem_buffer + i * SMEM_SIZE);
        smem_y[i] = reinterpret_cast<bf16*>(smem_buffer + NumStages * SMEM_SIZE + i * SMEM_SIZE);
    }

    auto barrier_ptr = reinterpret_cast<Barrier*>(smem_buffer + NumStages * SMEM_SIZE * 2);
#pragma unroll
    for (int i = 0; i < NumStages; i++) {
        full_barrier[i] = barrier_ptr + i;
        empty_barrier[i] = barrier_ptr + NumStages + i;
    }

    if (threadIdx.x == NumConsumers * 32) {
#pragma unroll
        for (int i = 0; i < NumStages; i++) {
            // Initialize with 1 because a single elected thread issues TMA operations.
            // TMA is dedicated hardware; its throughput does not depend on how many
            // threads attempt to issue the operation.
            full_barrier[i]->init(1);
            empty_barrier[i]->init(NumConsumers);
        }
        asm volatile("fence.proxy.async.shared::cta; \n" ::);
        asm volatile("fence.mbarrier_init.release.cluster; \n" ::);
    }
    __syncthreads();

    uint32_t warp_id = runtime::warpid();
    if (warp_id >= NumConsumers) {
        // Producer
        Scheduler<NumProducers, M, N, BlockM, BlockN, NumBlocks, NumStages> scheduler(warp_id - NumConsumers);
        uint32_t m_idx, n_idx;
        // Elect a single thread to perform wait/arrive and to issue TMA instructions
        if (not runtime::elect_one_sync()) {
            return;
        }
        while (scheduler.current_work_tile(m_idx, n_idx)) {
            empty_barrier[scheduler.stage_id]->wait((scheduler.current_iter + 1) & 1);
            // Load a BlockM x BlockN tile into shared memory via TMA.
            // One thread triggers the TMA instruction on dedicated hardware.
            // Using `cp.async.bulk` (not `cp.async.bulk.tensor`), so no tensor map is required,
            // but we are limited to 1D contiguous bytes per call.

            // We therefore issue all TMA operations from a single elected thread.
            // This also mirrors the common pattern with cp.async when comparing per-thread
            // issuance versus a single issuer.

            auto tma_async_load_v1 = [&]() {
                full_barrier[scheduler.stage_id]->arrive_and_expect_tx(BlockM * BlockN * sizeof(bf16));
                auto global_offset = (m_idx * BlockM) * N + n_idx * BlockN;
                auto* barrier_ptr = reinterpret_cast<uint64_t*>(full_barrier[scheduler.stage_id]);
                for (uint32_t i = 0; i < BlockM; i++) {
                    bf16* smem_x_ptr = smem_x[scheduler.stage_id] + i * BlockN;
                    async::TMA::load<1>(&smem_x_ptr[0], &x[global_offset + i * N], BlockN * sizeof(bf16), barrier_ptr);
                }
            };

            tma_async_load_v1();
            scheduler.step();
        }
    } else {
        // Consumer
        Scheduler<NumConsumers, M, N, BlockM, BlockN, NumBlocks, NumStages> scheduler(warp_id);
        uint32_t m_idx, n_idx;
        while (scheduler.current_work_tile(m_idx, n_idx)) {
            full_barrier[scheduler.stage_id]->wait(scheduler.current_iter & 1);
            // Compute using data resident in shared memory

            // Each thread processes 2 bf16 elements per register lane (NumElemPerThread = 4 bf16_2).
            // Compute y = 2 * x + 1 in registers, store results to shared memory,
            // then use TMA to store the results to global memory.
            // Note: routing through shared memory is slower than writing registers directly to global;
            // this path is used here to demonstrate TMA store.

            auto compute_and_store_v1 = [&]() {
                constexpr static uint32_t NumElemPerThread = 4;
                struct Register {
                    bf16_2 regs[NumElemPerThread];
                    __device__ __forceinline__ void load(void* src, uint32_t offs) {
                        using copy = memory::Move<bf16_2, memory::CopyAtom::OpS2R>;
                        void* src_ptr = reinterpret_cast<void*>(reinterpret_cast<bf16*>(src) + offs);
                        copy::move<NumElemPerThread>(regs, src_ptr);
                    }
                    __device__ __forceinline__ void compute() {
                        for (uint32_t i = 0; i < NumElemPerThread; i++) {
                            regs[i] = __hmul2(regs[i], __nv_bfloat162(2, 2));
                            regs[i] = __hadd2(regs[i], __nv_bfloat162(1, 1));
                        }
                    }
                    __device__ __forceinline__ void store(void* dst, uint32_t offs) {
                        // we need store the data to global memory directly
                        // cp.async just can help us to load data
                        using copy = memory::Move<bf16_2, memory::CopyAtom::OpR2S>;
                        void* dst_ptr = reinterpret_cast<void*>(reinterpret_cast<bf16*>(dst) + offs);
                        copy::move<NumElemPerThread>(dst_ptr, regs);
                    }
                } regs;
                for (int i = runtime::laneid() * NumElemPerThread * 2; i < BlockM * BlockN;
                     i += NumElemPerThread * 32 * 2) {
                    regs.load(smem_x[scheduler.stage_id], i);
                    regs.compute();
                    regs.store(smem_y[scheduler.stage_id], i);
                }

                if (runtime::elect_one_sync()) {
                    auto global_offset = (m_idx * BlockM) * N + n_idx * BlockN;
                    for (uint32_t i = 0; i < BlockM; i++) {
                        bf16* smem_y_ptr = smem_y[scheduler.stage_id] + i * BlockN;
                        async::TMA::store<1>(&out[global_offset + i * N], &smem_y_ptr[0], BlockN * sizeof(bf16));
                    }
                    async::TMA::commit_group();
                    async::TMA::store_async_wait();
                    empty_barrier[scheduler.stage_id]->arrive();
                }
            };

            compute_and_store_v1();
            scheduler.step();
        }
    }
}
} // namespace kernels::tma

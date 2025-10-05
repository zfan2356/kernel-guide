#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_bf16.h>
#include "prototype.cuh"
#include "tma/strategy.cuh"

#include <cutlass/arch/barrier.h>

namespace kernels::tma {

template <uint32_t M, uint32_t N, uint32_t BlockM, uint32_t BlockN, uint32_t NumBlocks,
    uint32_t NumStages>
struct Scheduler {
    explicit __device__ Scheduler() {
        num_m_blocks = utils::ceil_div(M, BlockM);
        num_n_blocks = utils::ceil_div(N, BlockN);
        num_blocks = num_m_blocks * num_n_blocks;
        current_block = blockIdx.x;
        current_iter = 0;
        stage_id = 0;
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
        current_block += NumBlocks;
        current_iter++;
        stage_id = (stage_id + 1) % NumStages;
    }

    uint32_t num_blocks, num_m_blocks, num_n_blocks;
    uint32_t current_block;
    int current_iter, stage_id;
};

template <uint32_t NumBlocks, uint32_t NumWarpsPerBlock, uint32_t NumConsumers,
    uint32_t NumProducers, uint32_t NumStages, uint32_t M, uint32_t N, uint32_t BlockM,
    uint32_t BlockN, uint32_t SwizzleMode>
__global__ __launch_bounds__(NumWarpsPerBlock * 32, 1) void tma_impl(bf16* x, bf16* out,
    const __grid_constant__ CUtensorMap tensor_map_x,
    const __grid_constant__ CUtensorMap tensor_map_out) {
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
    static_assert(NumConsumers + NumProducers == NumWarpsPerBlock);

    if (threadIdx.x == NumConsumers * 32) {
        async::TMA::prefetch_tensor_map(&tensor_map_x);
        async::TMA::prefetch_tensor_map(&tensor_map_out);
    }
    // DeepGemm use syncwarp to synchronize the threads, I don't know why
    __syncwarp();

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

    constexpr static uint32_t NumElemPerThread = 4;
    using tma_strategy = TmaStrategyV1<NumElemPerThread, BlockM, BlockN, M, N>;

    uint32_t warp_id = runtime::warpid();
    if (warp_id >= NumConsumers) {
        // Producer
        Scheduler<M, N, BlockM, BlockN, NumBlocks, NumStages> scheduler;
        uint32_t m_idx, n_idx;
        // Elect a single thread to perform wait/arrive and to issue TMA instructions
        // we can use elect one sync just because producer warp is one
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
            // full_barrier[scheduler.stage_id]->arrive();
            auto* barrier_ptr = reinterpret_cast<uint32_t*>(full_barrier[scheduler.stage_id]);
            tma_strategy::tma_async_load(
                smem_x[scheduler.stage_id], x, barrier_ptr, m_idx, n_idx, tensor_map_x);

            full_barrier[scheduler.stage_id]->arrive_and_expect_tx(BlockM * BlockN * sizeof(bf16));
            scheduler.step();
        }
    } else {
        // Consumer
        bool is_first_ok = false;
        bool is_first_wrong = false;
        Scheduler<M, N, BlockM, BlockN, NumBlocks, NumStages> scheduler;
        uint32_t m_idx, n_idx;
        while (scheduler.current_work_tile(m_idx, n_idx)) {
            full_barrier[scheduler.stage_id]->wait(scheduler.current_iter & 1);
            // if (runtime::elect_one_sync()) {
            //     for (int i = 0; i < BlockM; i++) {
            //         for (int j = 0; j < BlockN; j++) {
            //             float smem_x_val =
            //                 __bfloat162float(smem_x[scheduler.stage_id][i * BlockN + j]);
            //             float x_val =
            //                 __bfloat162float(x[(m_idx * BlockM + i) * N + n_idx * BlockN + j]);
            //             if (smem_x_val != x_val && !is_first_wrong) {
            //                 is_first_wrong = true;
            //                 printf("Error at %d, %d\n", i, j);
            //             } else if (smem_x_val == x_val && !is_first_ok) {
            //                 is_first_ok = true;
            //                 printf("Success at %d, %d\n", i, j);
            //             }
            //         }
            //     }
            // }
            // Compute using data resident in shared memory

            // Each thread processes 2 bf16 elements per register lane (NumElemPerThread = 4
            // bf16_2). Compute y = 2 * x + 1 in registers, store results to shared memory, then use
            // TMA to store the results to global memory. Note: routing through shared memory is
            // slower than writing registers directly to global; this path is used here to
            // demonstrate TMA store.
            tma_strategy::compute_and_store(smem_x[scheduler.stage_id], smem_y[scheduler.stage_id],
                out, m_idx, n_idx, tensor_map_out);
            // can use elect one sync just because consumer warp is one
            asm volatile("fence.proxy.async.shared::cta; \n" ::);
            if (runtime::elect_one_sync()) {
                empty_barrier[scheduler.stage_id]->arrive();
            }

            scheduler.step();
        }
    }
}
} // namespace kernels::tma

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_bf16.h>
#include "prototype.cuh"
#include "tma/strategy.cuh"
#include "utils/util.cuh"
#include <cutlass/arch/barrier.h>

namespace kernels::tma {

template <uint32_t M, uint32_t N, uint32_t BlockM, uint32_t BlockN, uint32_t kNumSMs,
    uint32_t kNumStages>
struct Scheduler {
    static constexpr uint32_t kNumMBlocks = utils::ceil_div(M, BlockM);
    static constexpr uint32_t kNumNBlocks = utils::ceil_div(N, BlockN);
    static constexpr uint32_t kTotalBlocks = kNumMBlocks * kNumNBlocks;

    __device__ Scheduler() : current_block_(blockIdx.x), stage_id_(0), phases_{} {
        // Initialize all phases to -1, except the first stage which starts at 0
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++i) {
            phases_[i] = -1;
        }
        phases_[0] = 0;
    }

    __device__ __forceinline__ bool get_current_tile(uint32_t& m_idx, uint32_t& n_idx) const {
        if (current_block_ >= kTotalBlocks) {
            return false;
        }
        m_idx = current_block_ / kNumNBlocks;
        n_idx = current_block_ % kNumNBlocks;
        return true;
    }

    __device__ __forceinline__ void advance() {
        current_block_ += kNumSMs;
        stage_id_ = (stage_id_ + 1) % kNumStages;
        phases_[stage_id_] = (phases_[stage_id_] + 1) & 1;
    }

    __device__ __forceinline__ uint32_t stage_id() const {
        return stage_id_;
    }
    __device__ __forceinline__ int phase(uint32_t stage) const {
        return phases_[stage];
    }

private:
    uint32_t current_block_;
    uint32_t stage_id_;
    int phases_[kNumStages];
};

template <uint32_t kNumSMs, uint32_t kNumWarpsPerBlock, uint32_t kNumConsumers, uint32_t kNumProducers,
    uint32_t kNumStages, uint32_t M, uint32_t N, uint32_t BlockM, uint32_t BlockN,
    uint32_t SwizzleMode>
__global__ __launch_bounds__(kNumWarpsPerBlock * 32, 1) void tma_impl(bf16* x, bf16* out,
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
    static_assert(kNumConsumers + kNumProducers == kNumWarpsPerBlock);
    constexpr static auto kNumConsumerThreads = kNumConsumers * 32;
    constexpr static auto kNumProducerThreads = kNumProducers * 32;

    if (threadIdx.x == kNumConsumerThreads) {
        async::TMA::prefetch_tensor_map(&tensor_map_x);
        async::TMA::prefetch_tensor_map(&tensor_map_out);
    }
    // thinking...
    __syncwarp();

    // Initialize multi-stage shared memory
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    constexpr static auto SMEM_SIZE = BlockM * BlockN * sizeof(bf16);
    static_assert(SMEM_SIZE % 1024 == 0, "SMEM_SIZE must be aligned to 1024");

    auto smem_x = utils::PattenVisitor([&](int i) {
        return reinterpret_cast<bf16*>(smem_buffer + i * SMEM_SIZE);
    });
    auto smem_y = utils::PattenVisitor([&](int i) {
        return reinterpret_cast<bf16*>(smem_buffer + kNumStages * SMEM_SIZE + i * SMEM_SIZE);
    });

    auto barrier_ptr = reinterpret_cast<Barrier*>(smem_buffer + kNumStages * SMEM_SIZE * 2);
    auto full_barrier = utils::PattenVisitor([&](int i) {
        return barrier_ptr + i;
    });
    auto empty_barrier = utils::PattenVisitor([&](int i) {
        return barrier_ptr + kNumStages + i;
    });

    if (threadIdx.x == kNumConsumerThreads) {
        #pragma unroll
        for (int i = 0; i < kNumStages; i++) {
            // Initialize with 1 because a single elected thread issues TMA operations.
            // TMA is dedicated hardware; its throughput does not depend on how many
            // threads attempt to issue the operation.
            full_barrier[i]->init(1);
            empty_barrier[i]->init(kNumConsumers);
        }
        // Make initialized barriers visible to the async proxy
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    using tma_strategy = TmaStrategyV2<4, BlockM, BlockN, M, N>;

    uint32_t warp_id = runtime::warpid(), lane_id = runtime::laneid();
    if (warp_id >= kNumConsumers) {
        // Producer
        Scheduler<M, N, BlockM, BlockN, kNumSMs, kNumStages> scheduler;
        uint32_t m_idx, n_idx;
        // Elect a single thread to perform wait/arrive operations and issue TMA instructions.
        // We use elect_one_sync here because there is only one producer warp.
        if (threadIdx.x < kNumConsumerThreads + 32 and runtime::elect_one_sync()) {
            while (scheduler.get_current_tile(m_idx, n_idx)) {
                const uint32_t s = scheduler.stage_id();
                const int current_phase = scheduler.phase(s);
                empty_barrier[s]->wait((current_phase + 1) & 1);

                // We therefore issue all TMA operations from a single elected thread.
                // This also mirrors the common pattern with cp.async when comparing per-thread
                // issuance versus a single issuer.
                auto* barrier_ptr = reinterpret_cast<uint32_t*>(full_barrier[s]);
                // Important: Pass &tensor_map_x (address) rather than tensor_map_x itself.
                // Taking the address later in tma_strategy::tma_async_load causes illegal memory
                // access. This is likely related to the prefetch_tma_descriptor call above.
                tma_strategy::tma_async_load(
                    smem_x[s], x, barrier_ptr,
                    n_idx, m_idx, &tensor_map_x
                );
                full_barrier[s]->arrive_and_expect_tx(BlockM * BlockN * sizeof(bf16));
                scheduler.advance();
            }
        }
    } else {
        // Consumer
        Scheduler<M, N, BlockM, BlockN, kNumSMs, kNumStages> scheduler;
        uint32_t m_idx, n_idx;
        while (scheduler.get_current_tile(m_idx, n_idx)) {
            const uint32_t s = scheduler.stage_id();
            const int current_phase = scheduler.phase(s);
            full_barrier[s]->wait(current_phase & 1);
            tma_strategy::compute_and_store(
                smem_x[s], smem_y[s], out,
                n_idx, m_idx, &tensor_map_out
            );
            // Use elect_one_sync here because there is only one consumer warp
            if (runtime::elect_one_sync()) {
                empty_barrier[s]->arrive();
            }
            scheduler.advance();
        }
    }
}
} // namespace kernels::tma

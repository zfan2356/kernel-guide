#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_bf16.h>
#include "prototype.cuh"

namespace kernels::tma {
using namespace kernels::prototype;
using bf16_2 = __nv_bfloat162;
using bf16 = __nv_bfloat16;

template <bool IsConsumer, uint32_t NumConsumers> class GetRank {
    __device__ __forceinline__ uint32_t operator()() {
        uint32_t warp_id = runtime::warpid();
        if constexpr (IsConsumer) {
            return warp_id;
        } else {
            return warp_id - NumConsumers;
        }
    }
};

template <uint32_t NumWorkers, uint32_t M, uint32_t N, uint32_t BlockM, uint32_t BlockN, uint32_t NumBlocks,
          uint32_t NumStages, class GetRank>
struct Scheduler {
    constexpr static auto NumWarps = NumWorkers * NumBlocks;

    explicit __device__ Scheduler() {
        num_m_blocks = utils::ceil_div(M, BlockM);
        num_n_blocks = utils::ceil_div(N, BlockN);
        num_blocks = num_m_blocks * num_n_blocks;
        current_block = blockIdx.x * NumWorkers + GetRank()();
        current_iter = 0;
        stage_id = GetRank()();
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
__global__ __launch_bounds__(NumWarpsPerBlock * 32, 1) void tma_impl(__nv_bfloat16* x, __nv_bfloat16* out) {
    /*
    x: [M, N] bfloat16
    out: [M, N] bfloat16
    This kernel will copy x into shared memory in block-wise, then apply x to 2 * x + 1, then copy the result to out.
    we will use tma to load/store data between shared memory and global memory.
    related documents:
    https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies-using-the-tensor-memory-accelerator-tma

    As you can see, TMA is essentially a superset of cp.async.
    We can use TMA to accomplish everything that cp.async can do.
    Here, we will first use TMA as a simple replacement for cp.async.
    we will use tma async load/store to implement producer-consumer pattern, use warp-specialize
    */
    constexpr static uint32_t NumWarps = NumWarpsPerBlock * NumBlocks;

    // first, we need init a multi-stage shared memory
    // we use __nv_bfloat162 to utilize the 32-bit register
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto smem_d = reinterpret_cast<__nv_bfloat16*>(smem_buffer);

    bf16* smem_x[NumStages];
    bf16* smem_y[NumStages];
    sync::Semaphore* empty_barrier[NumStages];
    sync::Semaphore* full_barrier[NumStages];
    constexpr static auto SMEM_SIZE = BlockM * BlockN;

#pragma unroll
    for (int i = 0; i < NumStages; i++) {
        smem_x[i] = reinterpret_cast<__nv_bfloat16*>(smem_buffer + i * SMEM_SIZE);
        smem_y[i] = reinterpret_cast<__nv_bfloat16*>(smem_buffer + NumStages * SMEM_SIZE + i * SMEM_SIZE);
    }
    auto barrier_ptr = reinterpret_cast<sync::Semaphore*>(smem_buffer + NumStages * SMEM_SIZE * 2);
#pragma unroll
    for (int i = 0; i < NumStages; i++) {
        empty_barrier[i] = barrier_ptr + i;
        full_barrier[i] = barrier_ptr + NumStages + i;
    }

    if (threadIdx.x == NumConsumers * 32) {
#pragma unroll
        for (int i = 0; i < NumStages; i++) {
            full_barrier[i]->init(0, 1);
            empty_barrier[i]->init(NumConsumers, 0);
        }
        asm volatile("fence.proxy.async.shared::cta; \n" ::);
        asm volatile("fence.mbarrier_init.release.cluster; \n" ::);
    }
    __syncthreads();

    // scheduler to schedule the work tile, divided by block-wise, you should set the

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
            using copy = memory::Move<bf16_2, memory::CopyAtom::OpR2G>;
            void* dst_ptr = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(dst) + offs);
            copy::move<NumElemPerThread>(dst_ptr, regs);
        }
    } regs;

    uint32_t warp_id = runtime::warpid();

    if (warp_id >= NumConsumers) {
        // consumer
        Scheduler<NumConsumers, M, N, BlockM, BlockN, NumBlocks, NumStages, GetRank<true, NumConsumers>> scheduler;
        uint32_t m_idx, n_idx;
        while (scheduler.current_work_tile(m_idx, n_idx)) {
            empty_barrier[scheduler.stage_id]->wait((scheduler.current_iter + 1) & 1);
            // now we should load block_m  * block_n data to shared memory which used by tma
            // tma is triggered by one thread, that issue a instruction to a specific hardware
            // because we only use cp.async.bulk, rather than cp.async.bulk.tensor, we don't need tensormap
            // but it restricts us to load only 1d data

            // so as you can see, we also can use one thread to issue all the tma instructions
            // in the code, we explore the difference between using one thread and use all threads which is same as
            // cp.async

            auto launch_tma_use_one_thread = [&]() {
                // there are two producers. and we use the elected one thread of the producer warpgroup to issue the tma
                // instructions
                for (uint32_t i = 0; i < BlockM; i++) {
                    async::TMA::load<1>(smem_x[scheduler.stage_id] + i * BlockN,
                                        x + (m_idx * BlockM + i) * N + (n_idx * BlockN), BlockN * sizeof(bf16),
                                        full_barrier[scheduler.stage_id]);
                }
            };
            full_barrier[scheduler.stage_id]->arrive_and_expect(BlockM * BlockN * sizeof(bf16));
            scheduler.step();
        }
    } else {
        // producer
        Scheduler<NumProducers, M, N, BlockM, BlockN, NumBlocks, NumStages, GetRank<false, NumProducers>> scheduler;
        using consumer = runtime::Group<0, NumConsumers>;
        uint32_t m_idx, n_idx;
        while (scheduler.current_work_tile(m_idx, n_idx)) {
            full_barrier[scheduler.stage_id]->wait(scheduler.current_iter & 1);
            // we will compute the data in shared memory

            // because NumElemPerThread is 4 bf16_2, so we need to load 2 plus bf16 per thread
            for (int i = runtime::laneid() * NumElemPerThread * 2; i < BlockM * BlockN;
                 i += NumElemPerThread * 32 * 2) {
                regs.load(smem_x[scheduler.stage_id], i);
                regs.compute();
                uint32_t glo_x = i / BlockN, glo_y = i % BlockN;
                uint32_t glo_offs = (m_idx * BlockM + glo_x) * N + (n_idx * BlockN + glo_y);
                regs.store(out, glo_offs);
            }
            if (runtime::elect_one_sync()) {
                empty_barrier[scheduler.stage_id]->arrive(1);
            }
        }
    }
}
} // namespace kernels::tma

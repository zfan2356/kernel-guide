#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_bf16.h>
#include "prototype.cuh"

namespace kernels::cpasync {
using namespace kernels::prototype;
using bf16_2 = __nv_bfloat162;

template <uint32_t NumBlocks, uint32_t NumWarpsPerBlock, uint32_t M, uint32_t N, uint32_t BlockM, uint32_t BlockN>
__global__ __launch_bounds__(NumWarpsPerBlock * 32, 1) void cp_async_impl(__nv_bfloat16* x, __nv_bfloat16* out) {
    /*
    x: [M, N] bfloat16
    out: [M, N] bfloat16
    This kernel will copy x into shared memory in block-wise, then apply x to 2 * x + 1, then copy the result to out.
    we will use cp.async to copy data from shared memory to global memory. and use ld.shared from shared memory to
    register. to demonstrate the performance for async load use cp.async, we will use multi-pipline to overlap the
    memory access and computation.

    related documents:
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async

    there are two ways that can use multi-stage to overlap load/store and computation
    1. use producer-consumer pattern, half of the warps will be producer, half of the warps will be consumer
    2. just prefetch the first tile of data, and then in each step, we will wait the previous step load data,
    then compute, and prefetch the next tile of data asynchronously
    we will implement both of the two ways, first, we will implement the prefetch way, it does'nt need mbarrier to
    synchronize the threads.
    */
    constexpr static uint32_t NumStages = 2;
    constexpr static uint32_t NumWarps = NumWarpsPerBlock * NumBlocks;

    // first, we need init a multi-stage shared memory
    // we use __nv_bfloat162 to utilize the 32-bit register
    __shared__ bf16_2 smem_x[2][NumWarpsPerBlock][BlockM * BlockN / 2];

    // scheduler to schedule the work tile, divided by block-wise
    // and also control the stage id and is_first_step
    struct Scheduler {
        explicit __device__ Scheduler() {
            num_m_blocks = utils::ceil_div(M, BlockM);
            num_n_blocks = utils::ceil_div(N, BlockN);
            num_blocks = num_m_blocks * num_n_blocks;
            current_block = blockIdx.x * NumWarpsPerBlock + runtime::warpid();
            stage_id = 0;
            is_first_step = true;
        }
        __device__ __forceinline__ bool current_work_tile(uint32_t& m_idx, uint32_t& n_idx) {
            if (current_block >= num_blocks) {
                return false;
            }
            m_idx = current_block / num_n_blocks;
            n_idx = current_block % num_n_blocks;
            return true;
        }
        __device__ __forceinline__ bool next_work_tile(uint32_t& m_idx, uint32_t& n_idx) {
            uint32_t next_block = current_block + NumWarps;
            if (next_block >= num_blocks) {
                return false;
            }
            m_idx = next_block / num_n_blocks;
            n_idx = next_block % num_n_blocks;
            return true;
        }
        __device__ __forceinline__ void step() {
            current_block += NumWarps;
            stage_id = (stage_id + 1) % NumStages;
            is_first_step = false;
        }
        __device__ __forceinline__ int next_stage() const {
            return (stage_id + 1) % NumStages;
        }
        __device__ __forceinline__ uint32_t global_offs(uint32_t m_idx, uint32_t n_idx) const {
            // row major
            return m_idx * N + n_idx;
        }

        uint32_t num_blocks, num_m_blocks, num_n_blocks;
        uint32_t current_block, stage_id;
        bool is_first_step;
    } scheduler;

    // it means how many elements can be loaded to each thread
    // because we use 2 packd bfloat16 to each thread, so we need divide by 2
    // when block is 16 * 16, it means 8 elements can be loaded to each thread
    // and we use 4 registers to hold the data
    // we use a fixed number of elements per thread, and then use unroll for loop to load data
    // the second way is more flexible, the first way leads to register overflow if block is too large
    constexpr static uint32_t NumElemPerThread = 4;
    // a struct for register to manage compute and load/store
    struct Register {
        bf16_2 regs[NumElemPerThread];
        __device__ __forceinline__ void load(void* src, uint32_t offs) {
            using copy = memory::Move<bf16_2, memory::CopyAtom::OpS2R>;
            void* src_ptr = reinterpret_cast<void*>(reinterpret_cast<bf16_2*>(src) + offs);
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

    // we use unroll for loop to load data
    // so we will fix NumElemPerThread, and then use for loop
    // the advantage is that we can control the number of registers per thread
    // and not use static_assert to make sure BlockN must divide by NumElemPerThread * 2
    auto launch_cp_async_v2 = [scheduler, x](uint32_t m, uint32_t n, int stage_id) {
        static_assert(BlockM * (BlockN / 2) % 32 == 0, "BlockM * BlockN must be divisible by 32");
        // In fact, this assert can be avoided here, but for convenience, we implement it this way for now.
        static_assert((BlockM * (BlockN / 2)) % (NumElemPerThread * 32) == 0,
            "BlockM * BlockN must be divisible by NumElemPerThread * 32");
        constexpr uint32_t bytes = NumElemPerThread * sizeof(bf16_2);
#pragma unroll
        for (uint32_t offset = runtime::laneid() * NumElemPerThread; offset < BlockM * BlockN / 2;
            offset += NumElemPerThread * 32) {
            uint32_t glo_x = offset / (BlockN / 2), glo_y = offset % (BlockN / 2) * 2;
            uint32_t global_offs = scheduler.global_offs(m * BlockM + glo_x, n * BlockN + glo_y);
            async::CpAsync::call<bytes, async::CpAsync::CacheOperator::OpCG>(
                smem_x[stage_id][runtime::warpid()] + offset, x + global_offs);
        }
        async::CpAsync::commit_group();
    };
    uint32_t m_idx, n_idx;
    uint32_t next_m_idx, next_n_idx;
    while (scheduler.current_work_tile(m_idx, n_idx)) {
#ifdef USE_SYNC_VERSION
        launch_cp_async_v2(m_idx, n_idx, scheduler.stage_id);
        async::CpAsync::wait_group<0>();
#else
        if (scheduler.is_first_step) {
            // we need prefetch the first tile data to shared memory
            // noticed that cp.async is thread-level instruction, so we need load data to each thread
            launch_cp_async_v2(m_idx, n_idx, scheduler.stage_id);
        }
        // wait the current tile data from shared memory
        // 0 means that wait all the async load instructions
        // in the docs says that there is no ordering guarantee between two cp.async operations
        // so, we need wait all the async load instructions
        async::CpAsync::wait_group<0>();

        if (scheduler.next_work_tile(next_m_idx, next_n_idx)) {
            // we can before compute the current tile data, prefetch the next tile data
            // use cp.async
            launch_cp_async_v2(next_m_idx, next_n_idx, scheduler.next_stage());
        }
        // load data from shared memory to register, now problem is warp-level
        // we need load data from shared memory to register
#endif
#pragma unroll
        for (uint32_t i = runtime::laneid() * NumElemPerThread; i < BlockM * BlockN / 2; i += NumElemPerThread * 32) {
            regs.load(smem_x[scheduler.stage_id][runtime::warpid()], i);
            regs.compute();
            uint32_t glo_x = i / (BlockN / 2), glo_y = i % (BlockN / 2) * 2;
            uint32_t global_offs = scheduler.global_offs(m_idx * BlockM + glo_x, n_idx * BlockN + glo_y);
            regs.store(out, global_offs);
        }
        scheduler.step();
    }
}
} // namespace kernels::cpasync

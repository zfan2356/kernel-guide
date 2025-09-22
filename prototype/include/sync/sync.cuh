#pragma once

#include <cuda_runtime.h>
#include "../runtime/runtime.cuh"

/*
    This file is used to define the synchronization primitives for the prototype.
    we will implement below synchronize tools:
    - mbarrier
    - named barrier
*/

namespace kernels::prototype::sync {

using ValueType = uint64_t;

// a universal barrier, e.g. bar/barrier
struct Barrier {
    uint32_t barrier_id;
    __device__ __forceinline__ Barrier(uint32_t _id) : barrier_id(_id) {}
    __device__ __forceinline__ Barrier operator[](int i) {
        return Barrier(barrier_id + i);
    }

    __device__ __forceinline__ void arrive() {}
};

// mbarrier, which support `expect_bytes`
struct MBarrier {
    __device__ __forceinline__ static void arrive(const ValueType* sem, uint32_t count) {
        uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&sem));
        asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n" ::"r"(mbar_ptr), "r"(count)
                     : "memory");
    }

    __device__ __forceinline__ static void wait(const ValueType* sem, uint32_t phase) {
        uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&sem));
        asm volatile("{\n"
                     ".reg .pred                P1;\n"
                     "LAB_WAIT:\n"
                     "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
                     "@P1                       bra.uni DONE;\n"
                     "bra.uni                   LAB_WAIT;\n"
                     "DONE:\n"
                     "}\n" ::"r"(mbar_ptr),
                     "r"(phase));
    }

    __device__ __forceinline__ static int test_wait(const ValueType* sem, uint32_t phase) {
        uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&sem));
        int result;
        asm volatile("{\n"
                     ".reg .pred P1;\n"
                     "mbarrier.test_wait.parity.shared::cta.b64 P1, [%1], %2;\n"
                     "selp.u32 %0,1,0,P1;"
                     "}\n"
                     : "=r"(result)
                     : "r"(mbar_ptr), "r"(phase));
        return result;
    }

    __device__ __forceinline__ static void arrive_and_wait(const ValueType* sem, uint32_t phase, uint32_t count = 1) {
        arrive(sem, count);
        wait(sem, phase);
    }

    __device__ __forceinline__ static void expect_bytes(const ValueType* sem, uint32_t bytes) {
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&sem));
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar_ptr), "r"(bytes));
    }
};

// a warp level semaphore, use one thread to init
struct Semaphore {
    __device__ __forceinline__ void init(uint32_t thread_count, uint32_t transaction_count) {
        Semaphore::init(&this->value, thread_count, transaction_count);
    }
    __device__ __forceinline__ static void init(const ValueType* smem, uint32_t thread_count,
                                                uint32_t transaction_count) {
        if (runtime::elect_one_sync()) {
            uint32_t sem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));

            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(sem_ptr),
                         "r"(thread_count + transaction_count));
        }
    }
    __device__ __forceinline__ static void destroy(const ValueType* smem) {
        if (runtime::elect_one_sync()) {
            uint32_t sem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));

            asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" ::"r"(sem_ptr));
        }
    }
    __device__ __forceinline__ void wait(uint32_t phase) {
        MBarrier::wait(&this->value, phase);
    }
    __device__ __forceinline__ void arrive(uint32_t count) {
        MBarrier::arrive(&this->value, count);
    }
    __device__ __forceinline__ void arrive_and_expect(uint32_t bytes) {
        MBarrier::expect_bytes(&this->value, bytes);
    }

private:
    ValueType value;
};


} // namespace kernels::prototype::sync

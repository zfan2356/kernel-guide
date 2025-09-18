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

// a warp level semaphore, use one thread to init
struct Semaphore {
    __device__ __forceinline__ Semaphore() {}
    __device__ __forceinline__ void init(uint32_t thread_count, uint32_t transaction_count) {
        if (runtime::elect_one_sync()) {
            uint32_t sem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(this));

            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(sem_ptr),
                         "r"(thread_count + transaction_count));
        }
    }

    __device__ __forceinline__ ~Semaphore() {
        if (runtime::elect_one_sync()) {
            uint32_t sem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(this));

            asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" ::"r"(sem_ptr));
        }
    }
    Semaphore(const Semaphore&) = delete;
    Semaphore(Semaphore&&) = delete;
    Semaphore& operator=(const Semaphore&) = delete;
    Semaphore& operator=(Semaphore&&) = delete;

private:
    uint64_t value;
};

template <int NumWarps> struct Barrier {
    uint32_t barrier_id;
    __device__ __forceinline__ Barrier(uint32_t _id) : barrier_id(_id) {}
    __device__ __forceinline__ Barrier operator[](int i) {
        return Barrier(barrier_id + i);
    }

    __device__ __forceinline__ void arrive() {}
};

template <int NumWarps> struct MBarrier {
    __device__ __forceinline__ static void arrive(Semaphore& sem, uint32_t count) {
        uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&sem));
        asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n" ::"r"(mbar_ptr), "r"(count)
                     : "memory");
    }

    __device__ __forceinline__ static void wait(Semaphore& sem, uint32_t phase) {
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

    __device__ __forceinline__ static int test_wait(Semaphore& sem, uint32_t phase) {
        uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&sem));
        int result;
        asm volatile("{\n"
                     ".reg .pred                P1;\n"
                     "mbarrier.test_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
                     "selp.u32 %0,1,0,P1;\n"
                     "}\n" ::"r"(mbar_ptr),
                     "r"(phase));
        return result;
    }

    __device__ __forceinline__ static void arrive_and_wait(Semaphore& sem, uint32_t phase, uint32_t count = 1) {
        arrive(sem, count);
        wait(sem, phase);
    }

    __device__ __forceinline__ static void expect_bytes(Semaphore& sem, uint32_t bytes) {
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&sem));

        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar_ptr), "r"(bytes));
    }
};

// Concept to constrain Bytes parameter for cp.async
// cp-size can only be 4, 8 and 16 bytes
template <uint32_t Bytes> concept ValidCpAsyncBytes = (Bytes == 4 || Bytes == 8 || Bytes == 16);

struct CpAsync {
    enum class CacheOperator {
        OpCA,
        OpCG,
    };
    template <uint32_t Bytes, CacheOperator Op>
    requires ValidCpAsyncBytes<Bytes> __device__ __forceinline__ static void call(void* dst, const void* src) {
        uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
        if constexpr (Op == CacheOperator::OpCG) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst_ptr), "l"(src), "n"(Bytes) : "memory");
        } else {
            asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(dst_ptr), "l"(src), "n"(Bytes) : "memory");
        }
    }

    __device__ __forceinline__ static void commit_group() {
        asm volatile("cp.async.commit_group ;");
    }

    template <uint32_t N = 0> __device__ __forceinline__ static void wait_group() {
        if constexpr (N == 0) {
            asm volatile("cp.async.wait_all ;");
        } else {
            asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
        }
    }
};


} // namespace kernels::prototype::sync

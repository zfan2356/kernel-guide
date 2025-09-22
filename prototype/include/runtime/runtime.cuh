#pragma once

namespace kernels::prototype::runtime {

__device__ __forceinline__ uint32_t laneid() {
    uint32_t lane_id;
    asm("mov.u32 %0, %laneid;" : "=r"(lane_id));
    return lane_id;
}

__device__ __forceinline__ int warpid() {
    return __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
}

__device__ __forceinline__ int warpgroud_id() {
    return __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
}

__device__ __forceinline__ uint32_t elect_one_sync() {
    // Elect a leader thread from a set of threads.
    // Syntax: elect.sync d|p, membermask;
    int32_t pred = 0;
    uint32_t laneid = 0;
    asm volatile("{\n"
                 ".reg .b32 %%rx;\n"
                 ".reg .pred %%px;\n"
                 "     elect.sync %%rx|%%px, %2;\n"
                 "@%%px mov.s32 %1, 1;\n"
                 "     mov.s32 %0, %%rx;\n"
                 "}\n"
                 : "+r"(laneid), "+r"(pred)
                 : "r"(0xFFFFFFFF));
    return pred;
}

template <typename T> __device__ __host__ __forceinline__ static constexpr T max(T a, T b) {
    return a > b ? a : b;
}

// use [warpsbegin, warpsend) to define the group
template <uint32_t WarpsBegin, uint32_t WarpsEnd> struct Group {
    constexpr static auto NumWarps = WarpsEnd - WarpsBegin;
    constexpr static auto GroupWarps = NumWarps;
    constexpr static auto GroupThreads = NumWarps * 32;
    __device__ __forceinline__ static int laneid() {
        return max(0u, threadIdx.x - WarpsBegin * 32) % GroupThreads;
    }
    __device__ __forceinline__ static int warpid() {
        return laneid() / 32;
    }

    __device__ __forceinline__ static void sync(int id) {
        asm volatile("bar.sync %0, %1;\n" ::"r"(id), "n"(GroupThreads));
    }
};

} // namespace kernels::prototype::runtime

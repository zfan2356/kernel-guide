#pragma once

#include <cuda_bf16.h>

namespace kernels::prototype::memory {

/*
    related documents:
      https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st

*/
enum class CopyAtom {
    OpS2R, // shared to register
    OpR2S, // register to shared
    OpR2G, // register to global
    OpG2R, // global to register
};

/*
    this file is used to define the memory movement primitives for the prototype.
    we will implement below memory movement primitives:
    - Move<__nv_bfloat16, CopyAtom::OpS2R>
    - Move<float, CopyAtom::OpS2R>
    - Move<__nv_bfloat16, CopyAtom::OpR2G>
    - Move<float, CopyAtom::OpR2G>

    =r means the register is a write only output register
    r means the register is a read only input register
    +r means the register is a read and write register
*/

template <typename T, CopyAtom Op> struct Move;

template <> struct Move<__nv_bfloat16, CopyAtom::OpS2R> {
    // use ld.shared
    template <uint32_t N = 1> __device__ __forceinline__ static void move(void* dst, void* src) {
        uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(src));
        char* dst_ptr = static_cast<char*>(dst);
        if constexpr (N == 1) {
            asm volatile("ld.shared.b16 %0, [%1];\n" : "=r"(*reinterpret_cast<uint32_t*>(dst)) : "r"(src_ptr));
        } else if constexpr (N == 2) {
            asm volatile("ld.shared.v2.b16 {%0, %1}, [%2];\n"
                         : "=r"(*reinterpret_cast<uint32_t*>(dst_ptr)), "=r"(*reinterpret_cast<uint32_t*>(dst_ptr + 2))
                         : "r"(src_ptr));
        } else if constexpr (N == 4) {
            asm volatile("ld.shared.v4.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(*reinterpret_cast<uint32_t*>(dst_ptr)), "=r"(*reinterpret_cast<uint32_t*>(dst_ptr + 2)),
                           "=r"(*reinterpret_cast<uint32_t*>(dst_ptr + 4)),
                           "=r"(*reinterpret_cast<uint32_t*>(dst_ptr + 6))
                         : "r"(src_ptr));
        }
    }
};

template <typename T>
    requires std::is_same_v<T, float> || std::is_same_v<T, __nv_bfloat162> struct Move<T, CopyAtom::OpS2R> {
    // use ld.shared
    template <uint32_t N = 1> __device__ __forceinline__ static void move(void* dst, void* src) {
        uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(src));
        char* dst_ptr = static_cast<char*>(dst);
        if constexpr (N == 1) {
            asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(*reinterpret_cast<uint32_t*>(dst_ptr)) : "r"(src_ptr));
        } else if constexpr (N == 2) {
            asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];\n"
                         : "=r"(*reinterpret_cast<uint32_t*>(dst_ptr)), "=r"(*reinterpret_cast<uint32_t*>(dst_ptr + 4))
                         : "r"(src_ptr));
        } else if constexpr (N == 4) {
            asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(*reinterpret_cast<uint32_t*>(dst_ptr)), "=r"(*reinterpret_cast<uint32_t*>(dst_ptr + 4)),
                           "=r"(*reinterpret_cast<uint32_t*>(dst_ptr + 8)),
                           "=r"(*reinterpret_cast<uint32_t*>(dst_ptr + 12))
                         : "r"(src_ptr));
        }
    }
};

template <typename T>
    requires std::is_same_v<T, float> || std::is_same_v<T, __nv_bfloat162> struct Move<T, CopyAtom::OpR2G> {
    // use st.global.cs
    template <uint32_t N = 1> __device__ __forceinline__ static void move(void* dst, void* src) {
        char* src_ptr = static_cast<char*>(src);
        if constexpr (N == 1) {
            asm volatile("st.weak.global.cs.b32 [%0], %1;"
                         :
                         : "l"(dst), "r"(*reinterpret_cast<const uint32_t*>(src_ptr))
                         : "memory");
        } else if constexpr (N == 2) {
            asm volatile("st.weak.global.cs.v2.b32 [%0], {%1, %2};"
                         :
                         : "l"(dst), "r"(*reinterpret_cast<const uint32_t*>(src_ptr)),
                           "r"(*reinterpret_cast<const uint32_t*>(src_ptr + 4))
                         : "memory");
        } else if constexpr (N == 4) {
            asm volatile("st.weak.global.cs.v4.b32 [%0], {%1, %2, %3, %4};"
                         :
                         : "l"(dst), "r"(*reinterpret_cast<const uint32_t*>(src_ptr)),
                           "r"(*reinterpret_cast<const uint32_t*>(src_ptr + 4)),
                           "r"(*reinterpret_cast<const uint32_t*>(src_ptr + 8)),
                           "r"(*reinterpret_cast<const uint32_t*>(src_ptr + 12))
                         : "memory");
        }
    }
};

template <typename T>
    requires std::is_same_v<T, float> || std::is_same_v<T, __nv_bfloat162> struct Move<T, CopyAtom::OpR2S> {
    // use st.shared
    template <uint32_t N = 1> __device__ __forceinline__ static void move(void* dst, void* src) {
        char* src_ptr = static_cast<char*>(src);
        auto dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
        if constexpr (N == 1) {
            asm volatile("st.shared.b32 [%0], %1;"
                         :
                         : "r"(dst_ptr), "r"(*reinterpret_cast<const uint32_t*>(src_ptr))
                         : "memory");
        } else if constexpr (N == 2) {
            asm volatile("st.shared.v2.b32 [%0], {%1, %2};"
                         :
                         : "r"(dst_ptr), "r"(*reinterpret_cast<const uint32_t*>(src_ptr)),
                           "r"(*reinterpret_cast<const uint32_t*>(src_ptr + 4))
                         : "memory");
        } else if constexpr (N == 4) {
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                         :
                         : "r"(dst_ptr), "r"(*reinterpret_cast<const uint32_t*>(src_ptr)),
                           "r"(*reinterpret_cast<const uint32_t*>(src_ptr + 4)),
                           "r"(*reinterpret_cast<const uint32_t*>(src_ptr + 8)),
                           "r"(*reinterpret_cast<const uint32_t*>(src_ptr + 12))
                         : "memory");
        }
    }
};


template <typename T> requires std::is_same_v<T, __nv_bfloat16> struct Move<T, CopyAtom::OpR2G> {
    template <uint32_t N = 1> __device__ __forceinline__ static void move(void* dst, void* src) {
        // maybe we can use Move<float> to implement this
        char* src_ptr = static_cast<char*>(src);
        if constexpr (N == 1) {
            asm volatile("st.weak.global.cs.b16 [%0], %1;"
                         :
                         : "l"(dst), "r"(*reinterpret_cast<const uint32_t*>(src_ptr))
                         : "memory");
        } else if constexpr (N % 2 == 0 and N <= 8) {
            using copy = Move<float, CopyAtom::OpR2G>;
            float* dst_ptr = reinterpret_cast<float*>(dst);
            copy::move<N / 2>(dst_ptr, src);
        }
    }
};


} // namespace kernels::prototype::memory

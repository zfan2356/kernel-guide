#pragma once

#include <pybind11/pybind11.h>
#include "../jitter/kernel_runtime.hpp"
#include "../jitter/handle.hpp"
#include "../jitter/compiler.hpp"
#include "ATen/Context.h"
#include "pybind11/cast.h"
#include "torch/python.h"
#include <cuda.h>

namespace kernels {

class TMARuntime final : public LaunchRuntime<TMARuntime> {
public:
    struct Args {
        LaunchArgs launch_args;
        uint32_t num_sms;
        uint32_t num_warps_per_block;
        uint32_t num_consumers, num_producers;
        uint32_t num_stages;
        uint32_t m, n, block_m, block_n;
        const torch::Tensor& x;
        const torch::Tensor& out;
        CUtensorMap tensor_map_x;
        CUtensorMap tensor_map_out;
        uint32_t swizzle_mode;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(
            R"(
    #include <tma/tma.cuh>

    using namespace kernels::tma;

    static void __instantiate_kernel() {{
        auto ptr = reinterpret_cast<void*>(&tma_impl<
        {}, {},
        {}, {}, {},
        {}, {}, {}, {}, {}
        >);
    }};
    )",
            args.num_sms, args.num_warps_per_block, args.num_consumers, args.num_producers,
            args.num_stages, args.m, args.n, args.block_m, args.block_n, args.swizzle_mode);
    }

    static void launch_impl(
        const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        K_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config, args.x.data_ptr(), args.out.data_ptr(),
            args.tensor_map_x, args.tensor_map_out));
    }
};

namespace tma {
    static CUtensorMapDataType aten_dtype_to_tensor_map_dtype(const at::ScalarType& dtype) {
        switch (dtype) {
        case torch::kInt: return CU_TENSOR_MAP_DATA_TYPE_INT32;
        case torch::kFloat: return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
        case torch::kBFloat16: return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
        case torch::kFloat8_e4m3fn: return CU_TENSOR_MAP_DATA_TYPE_UINT8;
        default: K_HOST_UNREACHABLE("Unsupported dtype");
        }
    }

    static CUtensorMapSwizzle mode_into_tensor_map_swizzle(const int& mode) {
        switch (mode) {
        case 0: return CU_TENSOR_MAP_SWIZZLE_NONE;
        case 16: return CU_TENSOR_MAP_SWIZZLE_NONE;
        case 32: return CU_TENSOR_MAP_SWIZZLE_32B;
        case 64: return CU_TENSOR_MAP_SWIZZLE_64B;
        case 128: return CU_TENSOR_MAP_SWIZZLE_128B;
        default: K_HOST_UNREACHABLE("Unsupported swizzling mode");
        }
    }


    static CUtensorMap create_tensor_map(const torch::Tensor& x, uint32_t block_m, uint32_t block_n,
        uint32_t m, uint32_t n, uint32_t swizzle_mode) {
        // create tensor map for tma async load / store
        // it is row major, shape is [m, n], smem_block is [block_m, block_n]
        uint32_t gmem_inner_dim = n, gmem_outer_dim = m;
        uint32_t smem_inner_dim = block_n, smem_outer_dim = block_m;
        uint32_t gmem_stride = x.stride(-2);

        const auto& elem_size = static_cast<int>(x.element_size());
        // Adjust shared-memory inner dimension when using swizzle. For swizzle sizes
        // (32B/64B/128B), the SMEM box inner dimension must equal swizzle_bytes / elem_size.
        const auto chosen_swizzle = mode_into_tensor_map_swizzle(swizzle_mode);
        if (chosen_swizzle != CU_TENSOR_MAP_SWIZZLE_NONE) {
            K_HOST_ASSERT(swizzle_mode % static_cast<uint32_t>(elem_size) == 0);
            const uint32_t swizzle_elems = swizzle_mode / static_cast<uint32_t>(elem_size);
            K_HOST_ASSERT(block_n % swizzle_elems == 0 &&
                          "block_n must be a multiple of swizzle width in elements");
            smem_inner_dim = swizzle_elems;
        }
        CUtensorMap map{};
        // notes the data type
        const cuuint64_t gmem_dims[2] = {
            static_cast<cuuint64_t>(gmem_inner_dim), static_cast<cuuint64_t>(gmem_outer_dim)};
        const cuuint32_t smem_dims[2] = {
            static_cast<cuuint32_t>(smem_inner_dim), static_cast<cuuint32_t>(smem_outer_dim)};
        const cuuint64_t gmem_strides[1] = {
            static_cast<cuuint64_t>(gmem_stride * elem_size),
        };
        const cuuint32_t elem_strides[2] = {1, 1};
        printf("Making TMA desc: global memory: %d %d, shared memory: %d %d, outer stride: %d, "
               "swizzle: %d, elem size: %d\n",
            gmem_inner_dim, gmem_outer_dim, smem_inner_dim, smem_outer_dim, gmem_stride,
            swizzle_mode, elem_size);

        K_CUDA_DRIVER_CHECK(cuTensorMapEncodeTiled(&map,
            aten_dtype_to_tensor_map_dtype(x.scalar_type()), 2, x.data_ptr(), gmem_dims,
            gmem_strides, smem_dims, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, chosen_swizzle,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        return map;
    }

    static void tma_test(const torch::Tensor& x, const torch::Tensor& out, uint32_t swizzle_mode) {
        // launch 70 blocks for persistent kernel, and 4 warps to form a warp group
        // which two of them are consumers, and two of them are producers
        uint32_t num_stages = 4, num_consumers = 1, num_producers = 1;
        uint32_t num_sms = 70, num_warps_per_block = num_consumers + num_producers;
        const uint32_t m = x.size(0), n = x.size(1);
        const uint32_t block_m = 64, block_n = 64;

        K_HOST_ASSERT(m % block_m == 0 && n % block_n == 0);
        K_HOST_ASSERT(m == out.size(0) && n == out.size(1));
        K_HOST_ASSERT(num_consumers + num_producers == num_warps_per_block);

        const auto& tensor_map_x = create_tensor_map(x, block_m, block_n, m, n, swizzle_mode);
        const auto& tensor_map_out = create_tensor_map(out, block_m, block_n, m, n, swizzle_mode);

        uint32_t smem_size = num_stages * block_m * block_n * 2 * 2 + 1024;
        const TMARuntime::Args args{
            .launch_args = LaunchArgs(num_sms, num_warps_per_block * 32, smem_size),
            .num_sms = num_sms,
            .num_warps_per_block = num_warps_per_block,
            .num_consumers = num_consumers,
            .num_producers = num_producers,
            .num_stages = num_stages,
            .m = m,
            .n = n,
            .block_m = block_m,
            .block_n = block_n,
            .x = x,
            .out = out,
            .tensor_map_x = tensor_map_x,
            .tensor_map_out = tensor_map_out,
            .swizzle_mode = swizzle_mode,
        };
        const auto& code = TMARuntime::generate(args);
        const auto& runtime = compiler->build("tma_test", code);
        TMARuntime::launch(runtime, args);
    }

    static void register_apis(pybind11::module_& m) {
        m.def("tma_test", &tma_test, pybind11::arg("x"), pybind11::arg("out"),
            pybind11::arg("swizzle_mode") = 16);
    }

} // namespace tma
} // namespace kernels

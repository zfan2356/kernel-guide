#pragma once

#include <pybind11/pybind11.h>
#include "../jitter/kernel_runtime.hpp"
#include "../jitter/handle.hpp"
#include "../jitter/compiler.hpp"
#include "ATen/Context.h"
#include "pybind11/cast.h"
#include "torch/python.h"

namespace kernels {

class CpAsyncRuntime final : public LaunchRuntime<CpAsyncRuntime> {
public:
    struct Args {
        LaunchArgs launch_args;
        uint32_t num_blocks;
        uint32_t num_warps_per_block;
        uint32_t m, n, block_m, block_n;
        const torch::Tensor& x;
        const torch::Tensor& out;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(
            R"(
    #include <cp_async/cp_async.cuh>

    using namespace kernels::cpasync;

    static void __instantiate_kernel() {{
        auto ptr = reinterpret_cast<void*>(&cp_async_impl<
        {}, {},
        {}, {}, {}, {}
        >);
    }};
    )",
            args.num_blocks, args.num_warps_per_block, args.m, args.n, args.block_m, args.block_n);
    }

    static void launch_impl(
        const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        K_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config, args.x.data_ptr(), args.out.data_ptr()));
    }
};

namespace cpasync {
    static void cp_async_test(const torch::Tensor& x, const torch::Tensor& out) {
        // launch 70 blocks for persistent kernel, and 4 warps to form a warp group to test cp async
        uint32_t num_blocks = 70, num_warps_per_block = 4;
        const uint32_t m = x.size(0), n = x.size(1);
        const uint32_t block_m = 16, block_n = 16;

        K_HOST_ASSERT(m % block_m == 0 && n % block_n == 0);
        K_HOST_ASSERT(m == out.size(0) && n == out.size(1));

        uint32_t smem_size = 2 * 16 * 16 * 2 * num_warps_per_block + 1024;
        const CpAsyncRuntime::Args args{
            .launch_args = LaunchArgs(num_blocks, num_warps_per_block * 32, smem_size),
            .num_blocks = num_blocks,
            .num_warps_per_block = num_warps_per_block,
            .m = m,
            .n = n,
            .block_m = block_m,
            .block_n = block_n,
            .x = x,
            .out = out,
        };
        const auto& code = CpAsyncRuntime::generate(args);
        const std::string ptx_path = "include/cp_async/cp_async.ptx";
        const auto& runtime = compiler->build("cp_async_test", code, ptx_path);
        CpAsyncRuntime::launch(runtime, args);
    }

    static void register_apis(pybind11::module_& m) {
        m.def("cp_async_test", &cp_async_test, pybind11::arg("x"), pybind11::arg("out"));
    }

} // namespace cpasync
} // namespace kernels

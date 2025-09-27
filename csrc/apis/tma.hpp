#pragma once

#include <pybind11/pybind11.h>
#include "../jitter/kernel_runtime.hpp"
#include "../jitter/handle.hpp"
#include "../jitter/compiler.hpp"
#include "ATen/Context.h"
#include "pybind11/cast.h"
#include "torch/python.h"

namespace kernels {

class TMARuntime final : public LaunchRuntime<TMARuntime> {
public:
    struct Args {
        LaunchArgs launch_args;
        uint32_t num_blocks;
        uint32_t num_warps_per_block;
        uint32_t num_consumers, num_producers;
        uint32_t num_stages;
        uint32_t m, n, block_m, block_n;
        const torch::Tensor& x;
        const torch::Tensor& out;
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
        {}, {}, {}, {}
        >);
    }};
    )",
            args.num_blocks, args.num_warps_per_block, args.num_consumers, args.num_producers, args.num_stages, args.m,
            args.n, args.block_m, args.block_n);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        K_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config, args.x.data_ptr(), args.out.data_ptr()));
    }
};

namespace tma {
    static void tma_test(const torch::Tensor& x, const torch::Tensor& out) {
        // launch 70 blocks for persistent kernel, and 4 warps to form a warp group
        // which two of them are consumers, and two of them are producers
        uint32_t num_stages = 4, num_consumers = 1, num_producers = 1;
        uint32_t num_blocks = 70, num_warps_per_block = num_consumers + num_producers;
        const uint32_t m = x.size(0), n = x.size(1);
        const uint32_t block_m = 64, block_n = 64;

        K_HOST_ASSERT(m % block_m == 0 && n % block_n == 0);
        K_HOST_ASSERT(m == out.size(0) && n == out.size(1));
        K_HOST_ASSERT(num_consumers + num_producers == num_warps_per_block);

        uint32_t smem_size = num_stages * block_m * block_n * 2 * 2 + 1024;
        const TMARuntime::Args args{
            .launch_args = LaunchArgs(num_blocks, num_warps_per_block * 32, smem_size),
            .num_blocks = num_blocks,
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
        };
        const auto& code = TMARuntime::generate(args);
        const auto& runtime = compiler->build("tma_test", code);
        TMARuntime::launch(runtime, args);
    }

    static void register_apis(pybind11::module_& m) {
        m.def("tma_test", &tma_test, pybind11::arg("x"), pybind11::arg("out"));
    }

} // namespace tma
} // namespace kernels

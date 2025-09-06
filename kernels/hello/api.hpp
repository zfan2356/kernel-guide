#pragma once

#include <pybind11/pybind11.h>
#include "../../jit/jitter/kernel_runtime.hpp"
#include "../../jit/jitter/handle.hpp"
#include "../../jit/jitter/compiler.hpp"

namespace kernels {

class HelloWorldRuntime final : public LaunchRuntime<HelloWorldRuntime> {
public:
    struct Args {
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(
            R"(
    #include <kernels/hello/hello.cuh>
    
    using namespace kernels::hello;
    
    static void __instantiate_kernel() {{
        auto ptr = reinterpret_cast<void*>(&hello_world_impl);
    }};
    )");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        K_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config));
    }
};

namespace hello {
static void hello_world() {
    const HelloWorldRuntime::Args args {
        .launch_args = LaunchArgs(1, 1),
    };
    const auto& code = HelloWorldRuntime::generate(args);
    const auto& runtime = compiler->build("hello_world", code);
    HelloWorldRuntime::launch(runtime, args);
}

static void register_apis(pybind11::module_& m) {
    m.def("hello_world", &hello_world);
}

} // namespace hello
} // namespace kernels
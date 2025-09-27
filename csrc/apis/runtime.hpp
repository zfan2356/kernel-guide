#pragma once

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <string>
#include "../jitter/compiler.hpp"
#include "../jitter/kernel_runtime.hpp"

namespace kernels {

namespace runtime {

    static void register_apis(pybind11::module_& m) {
        m.def(
            "init",
            [&](const std::string& library_root_path, const std::string& cuda_home_path_by_python,
                const std::vector<std::string>& custom_include_paths = {}) {
                Compiler::prepare_init(library_root_path, cuda_home_path_by_python, custom_include_paths);
                KernelRuntime::prepare_init(cuda_home_path_by_python);
            },
            pybind11::arg("library_root_path"), pybind11::arg("cuda_home_path_by_python"),
            pybind11::arg("custom_include_paths") = std::vector<std::string>{});
    }
} // namespace runtime

} // namespace kernels

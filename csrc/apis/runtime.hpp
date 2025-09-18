#pragma once

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <string>
#include "../jitter/compiler.hpp"
#include "../jitter/kernel_runtime.hpp"

namespace kernels {

namespace runtime {

    static void register_apis(pybind11::module_& m) {
        m.def("init", [&](const std::string& library_root_path, const std::string& cuda_home_path_by_python) {
            Compiler::prepare_init(library_root_path, cuda_home_path_by_python);
            KernelRuntime::prepare_init(cuda_home_path_by_python);
        });
    }
} // namespace runtime

} // namespace kernels

#include <pybind11/pybind11.h>
#include <torch/python.h>
#include <filesystem>

#include "apis/hello.hpp"
#include "apis/runtime.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME kernels_cpp 
#endif

// ReSharper disable once CppParameterMayBeConstPtrOrRef
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Kernels C++ library";

    kernels::runtime::register_apis(m);
    kernels::hello::register_apis(m);
}

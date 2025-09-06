#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "hello/api.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME kernels_cpp 
#endif

// ReSharper disable once CppParameterMayBeConstPtrOrRef
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Kernels C++ library";

    kernels::hello::register_apis(m);
}

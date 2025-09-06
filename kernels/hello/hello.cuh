#pragma once

#include <cstdio>

namespace kernels {
__global__ __launch_bounds__(1, 1) void hello_world_impl() {
    if (threadIdx.x == 0 and blockIdx.x == 0) {
        printf("Hello World\n");
    }
}
} // namespace kernels::hello
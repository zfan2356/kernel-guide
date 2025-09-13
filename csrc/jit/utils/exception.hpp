#pragma once

#include <exception>
#include <string>
#include <sstream>

namespace kernels {

class KernelException final : public std::exception {
    std::string message = {};

public:
    explicit KernelException(const char* name, const char* file, const int line, const std::string& error) {
        message = std::string(name) + " error (" + file + ":" + std::to_string(line) + "): " + error;
    }

    const char* what() const noexcept override {
        return message.c_str();
    }
};

#ifndef K_STATIC_ASSERT
#    define K_STATIC_ASSERT(cond, ...) static_assert(cond, __VA_ARGS__)
#endif

#ifndef K_HOST_ASSERT
#    define K_HOST_ASSERT(cond)                                                                                        \
        do {                                                                                                           \
            if (not(cond)) {                                                                                           \
                throw KernelException("Assertion", __FILE__, __LINE__, #cond);                                         \
            }                                                                                                          \
        } while (0)
#endif

#ifndef K_HOST_UNREACHABLE
#    define K_HOST_UNREACHABLE(reason) (throw KernelException("Assertion", __FILE__, __LINE__, reason))
#endif

#ifndef K_NVRTC_CHECK
#    define K_NVRTC_CHECK(cmd)                                                                                         \
        do {                                                                                                           \
            const auto& e = (cmd);                                                                                     \
            if (e != NVRTC_SUCCESS) {                                                                                  \
                throw KernelException("NVRTC", __FILE__, __LINE__, nvrtcGetErrorString(e));                            \
            }                                                                                                          \
        } while (0)
#endif

#ifndef K_CUDA_DRIVER_CHECK
#    define K_CUDA_DRIVER_CHECK(cmd)                                                                                   \
        do {                                                                                                           \
            const auto& e = (cmd);                                                                                     \
            if (e != CUDA_SUCCESS) {                                                                                   \
                std::stringstream ss;                                                                                  \
                const char *name, *info;                                                                               \
                cuGetErrorName(e, &name), cuGetErrorString(e, &info);                                                  \
                ss << static_cast<int>(e) << " (" << name << ", " << info << ")";                                      \
                throw KernelException("CUDA driver", __FILE__, __LINE__, ss.str());                                    \
            }                                                                                                          \
        } while (0)
#endif

#ifndef K_CUDA_RUNTIME_CHECK
#    define K_CUDA_RUNTIME_CHECK(cmd)                                                                                  \
        do {                                                                                                           \
            const auto& e = (cmd);                                                                                     \
            if (e != cudaSuccess) {                                                                                    \
                std::stringstream ss;                                                                                  \
                ss << static_cast<int>(e) << " (" << cudaGetErrorName(e) << ", " << cudaGetErrorString(e) << ")";      \
                throw KernelException("CUDA runtime", __FILE__, __LINE__, ss.str());                                   \
            }                                                                                                          \
        } while (0)
#endif

} // namespace kernels

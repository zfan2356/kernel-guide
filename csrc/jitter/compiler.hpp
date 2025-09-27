#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <nvrtc.h>
#include <regex>
#include <string>
#include <vector>

#include "../utils/exception.hpp"
#include "../utils/format.hpp"
#include "../utils/hash.hpp"
#include "../utils/lazy_init.hpp"
#include "../utils/system.hpp"
#include "cache.hpp"
#include "device_runtime.hpp"

namespace kernels {

class Compiler {
public:
    static std::filesystem::path library_root_path;
    static std::filesystem::path library_include_path;
    static std::vector<std::filesystem::path> custom_include_paths;
    static std::filesystem::path cuda_home;
    static std::string library_version;

    static std::string get_library_version() {
        std::stringstream ss;
        for (const auto& f : collect_files(library_include_path)) {
            std::ifstream in(f, std::ios::binary);
            ss << in.rdbuf();
        }
        return get_hex_digest(ss.str());
    }

    static void prepare_init(const std::string& library_root_path, const std::string& cuda_home_path_by_python,
                             const std::vector<std::string>& custom_include_paths = {}) {
        Compiler::library_root_path = library_root_path;
        if (!custom_include_paths.empty()) {
            // Use custom include paths
            for (const auto& path : custom_include_paths) {
                Compiler::custom_include_paths.push_back(path);
            }
        }
        // Use default include path
        Compiler::library_include_path = Compiler::library_root_path / "include";
        Compiler::cuda_home = cuda_home_path_by_python;
        Compiler::library_version = get_library_version();
    }

    std::string signature, flags;
    std::filesystem::path cache_dir_path;

    Compiler() {
        // Check `prepare_init`
        K_HOST_ASSERT(not library_root_path.empty());
        K_HOST_ASSERT(not library_include_path.empty());
        K_HOST_ASSERT(not cuda_home.empty());
        K_HOST_ASSERT(not library_version.empty());

        // Cache settings
        cache_dir_path = std::filesystem::path(get_env<std::string>("HOME")) / ".kernels";
        // The compiler flags applied to all derived compilers
        signature = "unknown-compiler";
        flags = fmt::format("-std=c++{} --diag-suppress=39,161,174,177,186,940 "
                            "--ptxas-options=--register-usage-level=10",
                            get_env<int>("DG_JIT_CPP_STANDARD", 20));
        flags += " --ptxas-options=--verbose";
        if (get_env("DG_JIT_WITH_LINEINFO", 0))
            flags += " -Xcompiler -rdynamic -lineinfo";
    }

    virtual ~Compiler() = default;

    std::filesystem::path make_tmp_dir() const {
        return make_dirs(cache_dir_path / "tmp");
    }

    std::filesystem::path get_tmp_file_path() const {
        return make_tmp_dir() / get_uuid();
    }

    void put(const std::filesystem::path& path, const std::string& data) const {
        const auto tmp_file_path = get_tmp_file_path();

        // Write into the temporary file
        std::ofstream out(tmp_file_path, std::ios::binary);
        K_HOST_ASSERT(out.write(data.data(), data.size()));
        out.close();

        // Atomically replace
        std::filesystem::rename(tmp_file_path, path);
    }

    std::shared_ptr<KernelRuntime> build(const std::string& name, const std::string& code) const {
        const auto kernel_signature = fmt::format("{}$${}$${}$${}$${}", name, library_version, signature, flags, code);
        const auto dir_path =
            cache_dir_path / "cache" / fmt::format("kernel.{}.{}", name, get_hex_digest(kernel_signature));

        // Hit the runtime cache
        if (const auto& runtime = kernel_runtime_cache->get(dir_path); runtime != nullptr) {
            printf("Hit the runtime cache: %s\n", dir_path.c_str());
            return runtime;
        }

        // Create the kernel directory
        make_dirs(dir_path);

        // Compile into a temporary CUBIN
        const auto tmp_cubin_path = get_tmp_file_path();
        compile(code, dir_path, tmp_cubin_path);

        // Replace into the cache directory
        make_dirs(dir_path);
        std::filesystem::rename(tmp_cubin_path, dir_path / "kernel.cubin");

        // Put into the runtime cache
        const auto& runtime = kernel_runtime_cache->get(dir_path);
        K_HOST_ASSERT(runtime != nullptr);
        return runtime;
    }

    virtual void compile(const std::string& code, const std::filesystem::path& dir_path,
                         const std::filesystem::path& cubin_path) const = 0;
};

// NOLINTNEXTLINE(misc-definitions-in-headers)
K_DECLARE_STATIC_VAR_IN_CLASS(Compiler, library_root_path);
// NOLINTNEXTLINE(misc-definitions-in-headers)
K_DECLARE_STATIC_VAR_IN_CLASS(Compiler, library_include_path);
// NOLINTNEXTLINE(misc-definitions-in-headers)
K_DECLARE_STATIC_VAR_IN_CLASS(Compiler, custom_include_paths);
// NOLINTNEXTLINE(misc-definitions-in-headers)
K_DECLARE_STATIC_VAR_IN_CLASS(Compiler, cuda_home);
// NOLINTNEXTLINE(misc-definitions-in-headers)
K_DECLARE_STATIC_VAR_IN_CLASS(Compiler, library_version);

class NVCCCompiler final : public Compiler {
    std::filesystem::path nvcc_path;

    std::pair<int, int> get_nvcc_version() const {
        K_HOST_ASSERT(std::filesystem::exists(nvcc_path));

        // Call the version command
        const auto& command = std::string(nvcc_path) + " --version";
        const auto& [return_code, output] = call_external_command(command);
        K_HOST_ASSERT(return_code == 0);

        // The version should be at least 12.3, for the best performance with 12.9
        int major, minor;
        std::smatch match;
        K_HOST_ASSERT(std::regex_search(output, match, std::regex(R"(release (\d+\.\d+))")));
        std::sscanf(match[1].str().c_str(), "%d.%d", &major, &minor);
        K_HOST_ASSERT((major > 12 or (major == 12 and minor >= 3)) and "NVCC version should be >= 12.3");
        if (major == 12 and minor < 9)
            printf("Warning: please use at least NVCC 12.9 for the best DeepGEMM performance\n");
        return {major, minor};
    }

public:
    NVCCCompiler() {
        // Override the compiler signature
        nvcc_path = cuda_home / "bin" / "nvcc";
        if (const auto& env_nvcc_path = get_env<std::string>("DG_JIT_NVCC_COMPILER"); not env_nvcc_path.empty()) {
            nvcc_path = env_nvcc_path;
        }
        const auto& [nvcc_major, nvcc_minor] = get_nvcc_version();
        signature = fmt::format("NVCC{}.{}", nvcc_major, nvcc_minor);

        std::string include_dirs;
        if (!custom_include_paths.empty()) {
            // Use custom include paths
            for (const auto& path : custom_include_paths) {
                include_dirs += fmt::format("-I{} ", path.string());
            }
        }
        // Use default include path
        include_dirs += fmt::format("-I{} ", library_include_path.string());

        // The override the compiler flags
        // Only NVCC >= 12.9 supports arch-specific family suffix
        const auto& arch = device_runtime->get_arch(false, nvcc_major > 12 or nvcc_minor >= 9);
        flags = fmt::format("{} {} --gpu-architecture=sm_{} "
                            "--compiler-options=-fPIC,-O3,-fconcepts,-Wno-deprecated-declarations,-Wno-abi "
                            "-cubin -O3 --expt-relaxed-constexpr --expt-extended-lambda",
                            flags, include_dirs, arch);
    }

    void compile(const std::string& code, const std::filesystem::path& dir_path,
                 const std::filesystem::path& cubin_path) const override {
        // Write the code into the cache directory
        const auto& code_path = dir_path / "kernel.cu";
        put(code_path, code);

        // Compile
        const auto& command =
            fmt::format("{} {} -o {} {}", nvcc_path.c_str(), code_path.c_str(), cubin_path.c_str(), flags);
        printf("Running NVCC command: %s\n", command.c_str());
        const auto& [return_code, output] = call_external_command(command);
        if (return_code != 0) {
            printf("NVCC compilation failed: %s\n", output.c_str());
            K_HOST_ASSERT(false and "NVCC compilation failed");
        }

        printf("%s", output.c_str());
    }
};

static auto compiler =
    LazyInit<Compiler>([]() -> std::shared_ptr<Compiler> { return std::make_shared<NVCCCompiler>(); });

} // namespace kernels

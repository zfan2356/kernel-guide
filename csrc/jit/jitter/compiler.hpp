#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <nvrtc.h>
#include <regex>
#include <string>

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

    static void prepare_init(const std::string& library_root_path, const std::string& cuda_home_path_by_python) {
        Compiler::library_root_path = library_root_path;
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
        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PTXAS_VERBOSE", 0))
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
        if (const auto& runtime = kernel_runtime_cache->get(dir_path); runtime != nullptr)
            return runtime;

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

K_DECLARE_STATIC_VAR_IN_CLASS(Compiler, library_root_path);
K_DECLARE_STATIC_VAR_IN_CLASS(Compiler, library_include_path);
K_DECLARE_STATIC_VAR_IN_CLASS(Compiler, cuda_home);
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
        if (const auto& env_nvcc_path = get_env<std::string>("DG_JIT_NVCC_COMPILER"); not env_nvcc_path.empty())
            nvcc_path = env_nvcc_path;
        const auto& [nvcc_major, nvcc_minor] = get_nvcc_version();
        signature = fmt::format("NVCC{}.{}", nvcc_major, nvcc_minor);

        // The override the compiler flags
        // Only NVCC >= 12.9 supports arch-specific family suffix
        const auto& arch = device_runtime->get_arch(false, nvcc_major > 12 or nvcc_minor >= 9);
        flags = fmt::format("{} -I{} --gpu-architecture=sm_{} "
                            "--compiler-options=-fPIC,-O3,-fconcepts,-Wno-deprecated-declarations,-Wno-abi "
                            "-cubin -O3 --expt-relaxed-constexpr --expt-extended-lambda",
                            flags, library_include_path.c_str(), arch);
    }

    void compile(const std::string& code, const std::filesystem::path& dir_path,
                 const std::filesystem::path& cubin_path) const override {
        // Write the code into the cache directory
        const auto& code_path = dir_path / "kernel.cu";
        put(code_path, code);

        // Compile
        const auto& command =
            fmt::format("{} {} -o {} {}", nvcc_path.c_str(), code_path.c_str(), cubin_path.c_str(), flags);
        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PRINT_COMPILER_COMMAND", 0))
            printf("Running NVCC command: %s\n", command.c_str());
        const auto& [return_code, output] = call_external_command(command);
        if (return_code != 0) {
            printf("NVCC compilation failed: %s\n", output.c_str());
            K_HOST_ASSERT(false and "NVCC compilation failed");
        }

        // Print PTXAS log
        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PTXAS_VERBOSE", 0))
            printf("%s", output.c_str());
    }
};

class NVRTCCompiler final : public Compiler {
public:
    NVRTCCompiler() {
        // Override the compiler signature
        int major, minor;
        K_NVRTC_CHECK(nvrtcVersion(&major, &minor));
        signature = fmt::format("NVRTC{}.{}", major, minor);
        K_HOST_ASSERT((major > 12 or (major == 12 and minor >= 3)) and "NVRTC version should be >= 12.3");

        // Build include directories list
        std::string include_dirs;
        include_dirs += fmt::format("-I{} ", library_include_path.string());
        include_dirs += fmt::format("-I{} ", (cuda_home / "include").string());

        // Add PCH support for version 12.8 and above
        // NOTES: PCH is vital for compilation speed
        std::string pch_flags;
        if (major > 12 or minor >= 8) {
            pch_flags = "--pch ";
            if (get_env<int>("DG_JIT_DEBUG", 0))
                pch_flags += "--pch-verbose=true ";
        }

        // Override the compiler flags
        // Only NVRTC >= 12.9 supports arch-specific family suffix
        const auto& arch = device_runtime->get_arch(false, major > 12 or minor >= 9);
        flags = fmt::format("{} {}--gpu-architecture=sm_{} -default-device {}", flags, include_dirs, arch, pch_flags);
    }

    void compile(const std::string& code, const std::filesystem::path& dir_path,
                 const std::filesystem::path& cubin_path) const override {
        // Write the code into the cache directory
        const auto& code_path = dir_path / "kernel.cu";
        put(code_path, code);

        // Parse compilation options
        std::istringstream iss(flags);
        std::vector<std::string> options;
        std::string option;
        while (iss >> option)
            options.push_back(option);

        // Convert to C-style string array for NVRTC
        std::vector<const char*> option_cstrs;
        for (const auto& opt : options)
            option_cstrs.push_back(opt.c_str());

        // Print compiler command if requested
        if (get_env<int>("DG_JIT_DEBUG", 0) or get_env<int>("DG_JIT_PRINT_COMPILER_COMMAND", 0)) {
            printf("Compiling JIT runtime with NVRTC options: ");
            for (const auto& opt : options)
                printf("%s ", opt.c_str());
            printf("\n");
        }

        // Create NVRTC program and compile
        nvrtcProgram program;
        K_NVRTC_CHECK(nvrtcCreateProgram(&program, code.c_str(), "kernel.cu", 0, nullptr, nullptr));
        const auto& compile_result =
            nvrtcCompileProgram(program, static_cast<int>(option_cstrs.size()), option_cstrs.data());

        // Get and print compiler log
        size_t log_size;
        K_NVRTC_CHECK(nvrtcGetProgramLogSize(program, &log_size));
        if (get_env<int>("DG_JIT_DEBUG", 0) or compile_result != NVRTC_SUCCESS) {
            if (compile_result != NVRTC_SUCCESS)
                K_HOST_ASSERT(log_size > 1);
            if (log_size > 1) {
                std::string compilation_log(log_size, '\0');
                K_NVRTC_CHECK(nvrtcGetProgramLog(program, compilation_log.data()));
                printf("NVRTC log: %s\n", compilation_log.c_str());
            }
        }

        // Get CUBIN size and data
        size_t cubin_size;
        K_NVRTC_CHECK(nvrtcGetCUBINSize(program, &cubin_size));
        std::string cubin_data(cubin_size, '\0');
        K_NVRTC_CHECK(nvrtcGetCUBIN(program, cubin_data.data()));

        // Write into the file system
        put(cubin_path, cubin_data);

        // Cleanup
        K_NVRTC_CHECK(nvrtcDestroyProgram(&program));
    }
};

static auto compiler = LazyInit<Compiler>([]() -> std::shared_ptr<Compiler> {
    if (get_env<int>("DG_JIT_USE_NVRTC", 0)) {
        return std::make_shared<NVRTCCompiler>();
    } else {
        return std::make_shared<NVCCCompiler>();
    }
});

} // namespace kernels

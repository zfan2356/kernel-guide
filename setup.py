import os
import shutil
import subprocess

import torch
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension

current_dir = os.path.dirname(os.path.realpath(__file__))
cxx_flags = [
    "-std=c++20",
    "-O3",
    "-fPIC",
    "-Wno-psabi",
    "-Wno-deprecated-declarations",
    f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
]
sources = ["csrc/python_api.cpp"]
build_include_dirs = [
    f"{CUDA_HOME}/include",
    f"{CUDA_HOME}/include/cccl",
    "prototype/include",
    "kernels/include",
    "3rd/fmt/include",
]
build_libraries = ["cuda", "cudart", "nvrtc"]
build_library_dirs = [f"{CUDA_HOME}/lib64", f"{CUDA_HOME}/lib64/stubs"]


class PostDevelopCommand(develop):
    """Handle development mode (pip install -e .)"""

    def run(self):
        super().run()
        self._make_prototype_symlinks()

    @staticmethod
    def _make_prototype_symlinks():
        """Create symlinks from prototype/include to kernels/include"""
        prototype_include_dir = os.path.join(current_dir, "prototype/include")
        kernels_include_dir = os.path.join(current_dir, "kernels/include")

        if not os.path.exists(prototype_include_dir):
            return

        for item in os.listdir(prototype_include_dir):
            src_item = os.path.join(prototype_include_dir, item)
            dst_item = os.path.join(kernels_include_dir, item)

            # Remove existing directory or symlink
            if os.path.exists(dst_item) or os.path.islink(dst_item):
                if os.path.islink(dst_item):
                    os.unlink(dst_item)
                elif os.path.isdir(dst_item):
                    shutil.rmtree(dst_item)
                else:
                    os.remove(dst_item)

            # Create symlink with relative path
            rel_path = os.path.relpath(src_item, kernels_include_dir)
            if os.path.isdir(src_item):
                os.symlink(rel_path, dst_item, target_is_directory=True)
                print(f"Created directory symlink: {dst_item} -> {rel_path}")
            else:
                os.symlink(rel_path, dst_item)
                print(f"Created file symlink: {dst_item} -> {rel_path}")


class CustomBuildPy(build_py):
    """Handle regular install (pip install .)"""

    def run(self):
        self._prepare_includes()
        super().run()

    def _prepare_includes(self):
        """Copy prototype/include contents to build directory"""
        build_include_dir = os.path.join(self.build_lib, "kernels/include")
        os.makedirs(build_include_dir, exist_ok=True)

        # Copy kernels/include contents first
        kernels_src = os.path.join(current_dir, "kernels/include")
        if os.path.exists(kernels_src):
            for item in os.listdir(kernels_src):
                src_item = os.path.join(kernels_src, item)
                dst_item = os.path.join(build_include_dir, item)
                if os.path.exists(dst_item):
                    (
                        shutil.rmtree(dst_item)
                        if os.path.isdir(dst_item)
                        else os.remove(dst_item)
                    )
                if os.path.isdir(src_item):
                    shutil.copytree(src_item, dst_item)
                else:
                    shutil.copy2(src_item, dst_item)

        # Copy prototype/include contents (both directories and files)
        prototype_src = os.path.join(current_dir, "prototype/include")
        if os.path.exists(prototype_src):
            for item in os.listdir(prototype_src):
                src_item = os.path.join(prototype_src, item)
                dst_item = os.path.join(build_include_dir, item)
                if os.path.exists(dst_item):
                    (
                        shutil.rmtree(dst_item)
                        if os.path.isdir(dst_item)
                        else os.remove(dst_item)
                    )
                if os.path.isdir(src_item):
                    shutil.copytree(src_item, dst_item)
                else:
                    shutil.copy2(src_item, dst_item)


if __name__ == "__main__":
    # noinspection PyBroadException
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        revision = "+" + subprocess.check_output(cmd).decode("ascii").rstrip()
    except:
        revision = ""

    setup(
        name="kernel_guide",
        version="1.0.0" + revision,
        packages=find_packages(".", exclude=["tests"]),
        package_data={
            "kernels": [
                "include/**/*",
            ],
            "prototype": [
                "include/**/*",
            ],
        },
        ext_modules=[
            CUDAExtension(
                name="kernels_cpp",
                sources=sources,
                include_dirs=build_include_dirs,
                libraries=build_libraries,
                library_dirs=build_library_dirs,
                extra_compile_args=cxx_flags,
            )
        ],
        zip_safe=False,
        cmdclass={
            "develop": PostDevelopCommand,
            "build_py": CustomBuildPy,
        },
    )

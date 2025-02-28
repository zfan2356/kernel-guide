import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_cuda_root():
    nvcc = os.popen("which nvcc").read().strip()
    cuda_root = os.path.dirname(os.path.dirname(nvcc))
    return os.path.abspath(cuda_root)


cuda_root = get_cuda_root()

setup(
    name="cuda-practice",
    version="0.0.1",
    packages=find_packages(exclude=["build", "dist", "docs", "tests"]),
    ext_modules=[
        CUDAExtension(
            "cuda_practice",
            sources=[
                "csrc/bind.cpp",
                "csrc/prtc/wmma/wmma.cu",
            ],
            extra_compile_args={
                "nvcc": [
                    # f"-I{os.path.dirname(__file__)}/3rd/ThunderKittens/include",  // requires cuda-12.3+
                    # f"-I{os.path.dirname(__file__)}/3rd/ThunderKittens/prototype",
                    # f"-I{os.path.dirname(__file__)}/3rd/cutlass/include",
                    # f"-I{os.path.dirname(__file__)}/3rd/cutlass/tools/util/include",
                    f"-arch=sm_80",  # A10开发机
                    "-std=c++20",
                    "-DTORCH_COMPILE",
                    "--use_fast_math",
                    "-DNDEBUG",
                    "-Xcompiler=-Wno-psabi",
                    "-Xcompiler=-fno-strict-aliasing",
                    "--expt-extended-lambda",
                    "--expt-relaxed-constexpr",
                    "-Xnvlink=--verbose",
                    "-O3",
                    "-Xptxas=--verbose",
                    "-Xptxas=--warn-on-spills",
                    "--threads",
                    "4",
                    "--keep",
                    "--generate-line-info",
                    "--source-in-ptx",
                ],
                "cxx": [
                    "-std=c++20",
                    "-O3",
                    "-g",
                    "-Wall",
                ],
            },
            extra_link_args=[
                f"-L{cuda_root}/lib/stubs",
                "-lcuda",
            ],
            libraries=["cuda"],
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(),
    },
)

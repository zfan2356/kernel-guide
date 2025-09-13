import os
import setuptools
import shutil
import subprocess
import torch
from setuptools import find_packages
from setuptools.command.build_py import build_py
from torch.utils.cpp_extension import CUDAExtension, CUDA_HOME

current_dir = os.path.dirname(os.path.realpath(__file__))
cxx_flags = ['-std=c++20', '-O3', '-fPIC', '-Wno-psabi', '-Wno-deprecated-declarations',
             f'-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}']
sources = ['csrc/python_api.cpp']
build_include_dirs = [
    f'{CUDA_HOME}/include',
    f'{CUDA_HOME}/include/cccl',
    'kernels/include',
    '3rd/fmt/include',
    'prototype',
]
build_libraries = ['cuda', 'cudart', 'nvrtc']
build_library_dirs = [
    f'{CUDA_HOME}/lib64',
    f'{CUDA_HOME}/lib64/stubs'
]

class CustomBuildPy(build_py):
    def run(self):
        # First, prepare the include directories
        self.prepare_includes()

        # Finally, run the regular build
        build_py.run(self)

    def prepare_includes(self):
        # Create kernels/include directory
        kernels_include_dst = os.path.join(self.build_lib, 'kernels/include')
        os.makedirs(kernels_include_dst, exist_ok=True)

        # Copy kernels/include and prototype contents to kernels/include
        for src_dir in ['kernels/include', 'prototype']:
            src_path = os.path.join(current_dir, src_dir)
            if os.path.exists(src_path):
                for item in os.listdir(src_path):
                    src_item = os.path.join(src_path, item)
                    dst_item = os.path.join(kernels_include_dst, item)
                    if os.path.exists(dst_item):
                        if os.path.isdir(dst_item):
                            shutil.rmtree(dst_item)
                        else:
                            os.remove(dst_item)
                    if os.path.isdir(src_item):
                        shutil.copytree(src_item, dst_item)
                    else:
                        shutil.copy2(src_item, dst_item)


if __name__ == '__main__':
    # noinspection PyBroadException
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except:
        revision = ''

    # noinspection PyTypeChecker
    setuptools.setup(
        name='kernel_guide',
        version='1.0.0' + revision,
        packages=find_packages('.'),
        package_data={
            'kernels': [
                'include/**/*',
            ]
        },
        ext_modules=[
            CUDAExtension(name='kernels_cpp',
                          sources=sources,
                          include_dirs=build_include_dirs,
                          libraries=build_libraries,
                          library_dirs=build_library_dirs,
                          extra_compile_args=cxx_flags)
        ],
        zip_safe=False,
        cmdclass={
            'build_py': CustomBuildPy,
        },
    )

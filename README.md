# CUDA PRACTICE

> cuda best practice &amp; notes

this is a general template for customizing some CUDA operators to integrate into your own training framwork to achieve acceleration effects.

choose the right cuda/torch version, the gpu used in this project is A10, so the highest version of nvidia driver is 12.2, you can choose the version that suits your device.

```
micromamba install -c conda-forge cuda-toolkit=12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## prepare

When commit and push code, pleace use `pre-commit` to ensure consistent code formatting style

```shell
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

use git-submodule to manage 3rd libs, such as `ThunderKittens` or `cutlass`
```shell
git submodules init
git submodules update
```

## pybind version

the first way to bind cuda kernels to python is using `pybind11`.
you can use `pybind11` to bind cuda kernels to python, and then import them in python to use.

Run `pip install -e .` to use the kernels in the `prtc` namespace. You can then import `prtc` from the tests directory to execute and validate these kernels.

The characteristic of using pybind to bind CUDA kernels for Python calls is that when you use `pip install -e .` for installation, CUDA will perform a compilation to generate a shared object (SO) file. Subsequently, every call will use this static SO file. If you modify the CUDA code, you need to re-run `pip install -e .` to recompile; otherwise, the changes will not take effect.

modify `setup.py` to include third-party libs, such as `ThunderKittens` or `cutlass`, but remember to match your CUDA version, otherwise there will be dependency issues, for example, the minimum cuda version for `ThunderKittens` is 12.3 or higher...

### install pybind11

```shell
git clone https://github.com/pybind/pybind11.git
cd pybind11
cmake -B build
cd build
make check -j 4
sudo make install
```

### find Torch

```shell
python -c 'import torch;print(torch.utils.cmake_prefix_path)'
export Torch_DIR="/path/to/your/torchdir"
```


## jit version

the second and recommended way to bind cuda kernels to python is using jit. it is also suitable for c++ projects.

When using JIT for dynamic compilation, we don't need to use `pip install -e.` to compile the CUDA kernel. Instead, we can directly call `call_jit` in Python to achieve dynamic compilation.

### code walkthrough

- `compiler`:

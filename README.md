# Kernel Guide

## Overall

This project implements CUDA kernels (maybe cpp or CuteDSL) to learn how to utilize advanced features such as `TMA`, `WGMMA`, and other cutting-edge techniques.

Additionally, this project includes simple kernels that are compiled to `PTX` to help understand PTX instructions.

project also provide a general template for customizing some CUDA operators to integrate into your own training/inference framwork to achieve acceleration effects.

## Use Way

```python
git submodule update --init --recursive
pip install -e .
python -m unittest tests/test_hello.py
```

## Envirenment

- CUDA 12.9
- python 3.12

## Kernel Lists

- [x] CP Async

- [ ] ld.matrix + MMA

- [ ] TMA Swizzle

- [ ] Hopper WGMMA

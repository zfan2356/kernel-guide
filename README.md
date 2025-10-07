# Kernel Guide

## Overall

This project implements CUDA kernels (maybe cpp or CuteDSL) to learn how to utilize advanced features such as `CP.Async`, `ld.matrix`, `MMA`, `TMA`, `WGMMA`, and other **cutting-edge** techniques.

Additionally, this project includes simple kernels that are compiled to `PTX` to help understand PTX instructions.

project also provide a general template for customizing some CUDA operators to integrate into your own training/inference framwork to achieve acceleration effects.

## Use Way

```python
git clone https://github.com/zfan2356/kernel-guide.git --recursive
pip install -e .
bash develop.sh
python tests/test_hello.py
```

## Envirenment

- Hopper GPU Arch: H20/H800...
- CUDA 12.9
- python 3.12

## Kernel Lists

### CPP Kernels

- [x] CP Async: 1D Asynchronize Load

```
cp.async;
cp.async.commit_group;
cp.async.wait_group;
```

- [ ] ld.matrix, st.matrix, MMA

- [x] TMA: 1D Asynchronous Load and Store

```
cp.async.bulk.mbarrier::complete_tx::bytes;
cp.async.bulk.bulk_group;
```

- [x] TMA 2D Asynchronous Load and Store

```
cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes
cp.async.bulk.tensor.2d.global.shared::cta.bulk_group
```

- [ ] TMA 2D Swizzle 16/32/64/128B

- [ ] Hopper WGMMA

### Cute DSL Kernels

- [ ] Flash Attention v2

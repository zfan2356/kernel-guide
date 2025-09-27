import unittest

import torch
import triton
import triton.language as tl

import kernels


@triton.jit
def _ref_kernel(
    x: tl.tensor, out: tl.tensor, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_m_blocks = tl.cdiv(x.shape[0], BLOCK_M)
    num_n_blocks = tl.cdiv(x.shape[1], BLOCK_N)
    num_blocks = num_m_blocks * num_n_blocks

    for block_id in tl.range(pid, num_blocks, 70):
        block_m = block_id % num_m_blocks
        block_n = block_id // num_m_blocks
        start_m = block_m * BLOCK_M
        start_n = block_n * BLOCK_N

        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        x = tl.load(x + offs_m[:, None] * x.stride(0) + offs_n[None, :] * x.stride(1))
        tl.store(
            out + offs_m[:, None] * out.stride(0) + offs_n[None, :] * out.stride(1),
            x * 2 + 1,
        )


def ref_kernel(x: torch.Tensor, out: torch.Tensor):
    BLOCK_M = 16
    BLOCK_N = 16
    _ref_kernel[(70,)](x, out, BLOCK_M, BLOCK_N)


class TestCpAsync(unittest.TestCase):
    def test_cp_async(self):
        x = torch.randn((8192, 8192), dtype=torch.bfloat16, device="cuda") * 0.01

        out = torch.empty_like(x)

        kernels.cp_async_test(x, out)
        torch.cuda.synchronize()

        ref_out = x * 2 + 1
        torch.cuda.synchronize()

        torch.testing.assert_close(out, ref_out)


if __name__ == "__main__":
    unittest.main()

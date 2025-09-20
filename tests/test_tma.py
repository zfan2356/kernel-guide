import unittest

import torch
import triton
import triton.language as tl

import kernels


class TestTMA(unittest.TestCase):
    def test_tma(self):
        x = torch.randn((8192, 8192), dtype=torch.bfloat16, device="cuda") * 0.01

        out = torch.empty_like(x)

        kernels.tma_test(x, out)
        torch.cuda.synchronize()

        ref_out = x * 2 + 1
        torch.cuda.synchronize()

        torch.testing.assert_close(out, ref_out)

        nbytes = x.numel() * x.element_size() * 2
        t = triton.testing.do_bench(
            lambda: kernels.tma_test(x, out), warmup=10, rep=100
        )

        print(f"Performance: {nbytes * 1e-9 / t:.2f} TB/s")


if __name__ == "__main__":
    unittest.main()

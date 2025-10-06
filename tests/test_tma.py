import unittest

import torch
from triton.testing import do_bench

import kernels


class TestTMA(unittest.TestCase):
    def test_tma(self):
        x = torch.randn((2048, 2048), dtype=torch.bfloat16, device="cuda") * 0.01

        out = torch.empty_like(x)

        kernels.tma_test(x, out, 0)
        torch.cuda.synchronize()

        ref_out = x * 2 + 1
        torch.cuda.synchronize()
        torch.testing.assert_close(out, ref_out)

        t = do_bench(lambda: kernels.tma_test(x, out, 0))
        print(f"time: {t:.5f} ms")


if __name__ == "__main__":
    unittest.main()

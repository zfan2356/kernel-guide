import unittest

import torch
from triton.testing import do_bench

import kernels


class TestSwizzle(unittest.TestCase):
    def test_swizzle(self):
        m, n = 8192, 4096
        x = torch.randn((m, n), dtype=torch.bfloat16, device="cuda") * 0.01
        out = torch.empty_like(x)

        # 16B swizzle alse means non swizzle
        # kernels.tma_test(x, out, 16)
        # torch.cuda.synchronize()

        ref_out = x * 2 + 1
        # torch.cuda.synchronize()
        # torch.testing.assert_close(out, ref_out)
        swizzle_x_mode = 128
        swizzle_out_mode = 0

        kernels.tma_test(x, out, swizzle_x_mode, swizzle_out_mode)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, ref_out)

        t = do_bench(lambda: kernels.tma_test(x, out, swizzle_x_mode, swizzle_out_mode))
        print(f"time: {t:.5f} ms")


if __name__ == "__main__":
    unittest.main()

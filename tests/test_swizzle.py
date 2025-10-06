import unittest

import torch
from triton.testing import do_bench

import kernels


class TestSwizzle(unittest.TestCase):
    def test_swizzle(self):
        x = torch.randn((2048, 2048), dtype=torch.bfloat16, device="cuda") * 0.01

        out = torch.empty_like(x)

        # 16B swizzle alse means non swizzle
        # kernels.tma_test(x, out, 16)
        # torch.cuda.synchronize()

        ref_out = x * 2 + 1
        # torch.cuda.synchronize()
        # torch.testing.assert_close(out, ref_out)

        kernels.tma_test(x, out, 32)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, ref_out)

        # kernels.tma_test(x, out, 64)
        # torch.cuda.synchronize()
        # torch.testing.assert_close(out, ref_out)

        # kernels.tma_test(x, out, 128)
        # torch.cuda.synchronize()
        # torch.testing.assert_close(out, ref_out)


if __name__ == "__main__":
    unittest.main()

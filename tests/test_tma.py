import unittest

import torch

import kernels


class TestTMA(unittest.TestCase):
    def test_tma(self):
        x = torch.randn((256, 256), dtype=torch.bfloat16, device="cuda") * 0.01

        out = torch.empty_like(x)

        kernels.tma_test(x, out)
        torch.cuda.synchronize()

        ref_out = x * 2 + 1
        torch.cuda.synchronize()

        torch.testing.assert_close(out, ref_out)


if __name__ == "__main__":
    unittest.main()

import unittest

import torch

import kernels


class TestTMA(unittest.TestCase):
    def test_tma(self):
        x = torch.randn((2048, 2048), dtype=torch.bfloat16, device="cuda") * 0.01

        out = torch.empty_like(x)

        kernels.tma_test(x, out, 0)
        torch.cuda.synchronize()

        ref_out = x * 2 + 1
        torch.cuda.synchronize()

        for i in range(0, 2048, 128):
            for j in range(0, 2048, 128):
                try:
                    torch.testing.assert_close(
                        out[i : i + 128, j : j + 128], ref_out[i : i + 128, j : j + 128]
                    )
                except Exception as e:
                    print(f"Error at {i}, {j}")
                    raise e

        torch.testing.assert_close(out, ref_out)


if __name__ == "__main__":
    unittest.main()

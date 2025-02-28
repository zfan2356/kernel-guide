import time
import unittest

import torch
from parameterized import parameterized

from triton_impl.gemm import gemm_nt


def calculate_tflops(m, n, k, time_ms):
    """Calculate TFLOPS for GEMM operation.

    Args:
        m: Number of rows in matrix A
        n: Number of columns in matrix B
        k: Number of columns in matrix A (rows in matrix B)
        time_ms: Execution time in milliseconds

    Returns:
        float: TFLOPS value
    """
    # Total FLOPs = 2 * M * N * K
    total_flops = 2.0 * m * n * k
    # Convert time to seconds
    time_sec = time_ms / 1000.0
    # Calculate TFLOPS
    tflops = (total_flops / time_sec) / 1e12
    return tflops


class TestGemm(unittest.TestCase):
    @parameterized.expand(
        [
            (512, 512, 512),
        ]
    )
    def test_gemm_performance(self, m, n, k):
        """Test GEMM performance in TFLOPS."""
        # Create input matrices
        a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(k, n, dtype=torch.bfloat16, device="cuda")
        c = torch.zeros(m, n, dtype=torch.bfloat16, device="cuda")

        # Warm up
        for _ in range(10):
            c = gemm_nt(a, b, 132)

        # Synchronize CUDA
        torch.cuda.synchronize()

        # Measure execution time
        start_time = time.time()
        for _ in range(100):  # Run multiple times for better measurement
            c = gemm_nt(a, b, 132)
        torch.cuda.synchronize()
        end_time = time.time()

        # Calculate average time per operation (in milliseconds)
        avg_time_ms = (end_time - start_time) * 1000 / 100

        # Calculate TFLOPS
        tflops = calculate_tflops(m, n, k, avg_time_ms)

        print(f"\nMatrix size: {m}x{k} * {k}x{n}")
        print(f"Average execution time: {avg_time_ms:.2f} ms")
        print(f"Performance: {tflops:.2f} TFLOPS")

        # Verify correctness
        c_ref = a @ b
        torch.testing.assert_close(c, c_ref, atol=1e-2, rtol=1e-2)

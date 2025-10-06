import argparse
import time
from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import torch
from cutlass.cute.runtime import from_dlpack


class SGemmNT:
    def __init__(
        self,
        tile_shape_nmk: Tuple[int, int, int] = (128, 128, 8),
        num_stages: int = 3,
        num_threads: int = 256,
    ):
        self.tile_shape_nmk = tile_shape_nmk
        self.num_stages = num_stages
        self.num_threads = num_threads
        assert num_threads > 0, "needs at least one thread"
        assert num_threads % 16 == 0, "multiples of 16 required for MMA thread layout"

        self.block_m, self.block_n, self.block_k = self.tile_shape_nmk
        assert self.block_m % 16 == 0, "multiple of 16 required for tile dimension M"
        assert self.block_n % 16 == 0, "multiple of 16 required for tile dimension N"

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        self.a_major_mode = utils.LayoutEnum.from_tensor(mA)
        self.b_major_mode = utils.LayoutEnum.from_tensor(mB)
        self.c_major_mode = utils.LayoutEnum.from_tensor(mC)

        sA_layout = cute.make_layout(
            (self.block_m, self.block_k, self.num_stages),
            stride=(self.block_k, 1, self.block_k * self.block_m),
        )
        sB_layout = cute.make_layout(
            (self.block_n, self.block_k, self.num_stages),
            stride=(self.block_k, 1, self.block_k * self.block_n),
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Create tiled copy for A and B
        # use cp.async, vectorized copy
        # ///////////////////////////////////////////////////////////////////////////////
        num_vec = 4
        major_mode_size = self.block_k // num_vec
        tA = cute.make_layout(
            (self.num_threads // major_mode_size, major_mode_size),
            stride=(major_mode_size, 1),
        )
        tB = cute.make_layout(
            (self.num_threads // major_mode_size, major_mode_size),
            stride=(major_mode_size, 1),
        )
        vA = cute.make_layout((num_vec, 1))
        vB = cute.make_layout((num_vec, 1))

        atom_async_copy_A = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mA.element_type,
            num_bits_per_copy=mA.element_type.width * num_vec,
        )
        atom_async_copy_B = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mA.element_type,
            num_bits_per_copy=mB.element_type.width * num_vec,
        )

        tiled_copy_A = cute.make_tiled_copy_tv(atom_async_copy_A, tA, vA)
        tiled_copy_B = cute.make_tiled_copy_tv(atom_async_copy_B, tB, vB)

        # ///////////////////////////////////////////////////////////////////////////////
        # Create tiled MMA
        # ///////////////////////////////////////////////////////////////////////////////
        atoms_layout = cute.make_layout(
            (self.num_threads // 16, 16, 1), stride=(16, 1, 0)
        )
        op = cute.nvgpu.MmaUniversalOp(cutlass.Float32)


def run(
    mnk: Tuple[int, int, int],
    **kwargs,
):
    print(f"Running Ampere SIMT GEMM example:")
    print(f"mnk: {mnk}")
    M, N, K = mnk

    # Create and permute tensor A/B/C
    # A is [M, K], row major
    # B is [N, K], row major
    # C is [M, N], row major
    def create_and_permute_tensor(mode0, mode1, dtype):
        # (mode1, mode0) -> (mode0, mode1)
        shape = (mode0, mode1)
        return (
            torch.empty(*shape, dtype=torch.int32).random_(-5, 5).to(dtype=dtype).cuda()
        )

    a = create_and_permute_tensor(M, K, torch.float32)
    b = create_and_permute_tensor(N, K, torch.float32)
    c = create_and_permute_tensor(M, N, torch.float32)

    print(
        f"a: {a.shape, a.stride()}, b: {b.shape, b.stride()}, c: {c.shape, c.stride()}"
    )
    print(f"{a.dim_order()}, {b.dim_order()}, {c.dim_order()}")

    def convert_from_dlpack(
        x: torch.Tensor, leading_dim: int, alignment: int = 16, divisibility: int = 1
    ) -> cute.Tensor:
        return (
            from_dlpack(x, assumed_align=alignment)
            .mark_layout_dynamic(leading_dim=leading_dim)
            .mark_compact_shape_dynamic(
                mode=leading_dim,
                stride_order=x.dim_order(),
                divisibility=divisibility,
            )
        )

    gemm_dtype = cutlass.Float32
    a_tensor, b_tensor, c_tensor = [
        convert_from_dlpack(x, 1, 16, 128 // gemm_dtype.width) for x in [a, b, c]
    ]

    sgemm = SGemmNT()
    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)
    compiled_fn = cute.compile(
        sgemm,
        a_tensor,
        b_tensor,
        c_tensor,
        stream=current_stream,
    )

    print("Executing GEMM kernel...")
    compiled_fn(a_tensor, b_tensor, c_tensor)
    torch.cuda.synchronize()
    print("Verifying results...")
    ref = torch.einsum("mk,nk->mn", a, b)
    torch.testing.assert_close(c.cpu(), ref.cpu(), atol=1e-03, rtol=1e-05)
    print("Results verified successfully!")

    # def generate_tensors():
    #     # Create new tensors for each workspace to ensure cold L2 cache
    #     a_workspace = create_and_permute_tensor(M, K, a_major == "m", torch.float32)
    #     b_workspace = create_and_permute_tensor(N, K, b_major == "n", torch.float32)
    #     c_workspace = create_and_permute_tensor(M, N, c_major == "m", torch.float32)

    #     if static_shape:
    #         a_tensor_workspace = (
    #             from_dlpack(a_workspace, assumed_align=16)
    #             .mark_layout_dynamic(leading_dim=(1 if a_major == "k" else 0))
    #             .mark_compact_shape_dynamic(
    #                 mode=(1 if a_major == "k" else 0),
    #                 divisibility=divisibility_a,
    #             )
    #         )
    #     else:
    #         a_tensor_workspace = from_dlpack(a_workspace, assumed_align=16)

    #     b_tensor_workspace = (
    #         from_dlpack(b_workspace, assumed_align=16)
    #         .mark_layout_dynamic(leading_dim=(1 if b_major == "k" else 0))
    #         .mark_compact_shape_dynamic(
    #             mode=(1 if b_major == "k" else 0),
    #             divisibility=divisibility_b,
    #         )
    #     )

    #     c_tensor_workspace = (
    #         from_dlpack(c_workspace, assumed_align=16)
    #         .mark_layout_dynamic(leading_dim=(1 if c_major == "n" else 0))
    #         .mark_compact_shape_dynamic(
    #             mode=(1 if c_major == "n" else 0),
    #             divisibility=divisibility_c,
    #         )
    #     )

    #     return testing.JitArguments(
    #         a_tensor_workspace, b_tensor_workspace, c_tensor_workspace, current_stream
    #     )

    # workspace_count = 1
    # if use_cold_l2:
    #     one_workspace_bytes = (
    #         a.numel() * a.element_size()
    #         + b.numel() * b.element_size()
    #         + c.numel() * c.element_size()
    #     )
    #     workspace_count = testing.get_workspace_count(
    #         one_workspace_bytes, warmup_iterations, iterations
    #     )

    # avg_time_us = testing.benchmark(
    #     compiled_fn,
    #     workspace_generator=generate_tensors,
    #     workspace_count=workspace_count,
    #     stream=current_stream,
    #     warmup_iterations=warmup_iterations,
    #     iterations=iterations,
    # )

    # # Print execution results
    # print(f"Kernel execution time: {avg_time_us / 1e3:.4f} ms")

    # return avg_time_us  # Return execution time in microseconds


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str) -> Tuple[int, ...]:
        try:
            return tuple(int(x.strip()) for x in s.split(","))
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid format. Expected comma-separated integers."
            )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mnk", type=parse_comma_separated_ints, default=(256, 256, 64)
    )
    args = parser.parse_args()
    print("Running SIMT GEMM example:")

    torch.manual_seed(1024)

    run(args.mnk)
    print("PASS")

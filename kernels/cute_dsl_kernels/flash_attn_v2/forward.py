from typing import Callable, Optional, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import torch
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack


class FlashAttentionV2Forward:
    def __init__(
        self,
        head_dim: int,
        m_block_size: int = 128,
        n_block_size: int = 128,
        num_threads: int = 128,
        is_causal: bool = False,
    ):
        self._head_dim = head_dim
        self._m_block_size = m_block_size
        self._n_block_size = n_block_size
        self._num_threads = num_threads
        self._is_causal = is_causal

        # padding head_dim to a multiple of 32 as k_block_size
        self._head_dim_padded = (head_dim + 31) // 32 * 32

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        softmax_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(
            not (
                mQ.element_type == mK.element_type == mV.element_type == mO.element_type
            )
        ):
            raise TypeError("All tensors must have the same data type")
        self._dtype: Type[cutlass.Numeric] = mQ.element_type
        smem_k_block_size = 64 if self._head_dim_padded % 64 == 0 else 32
        swizzle_bits = 3 if smem_k_block_size == 64 else 2
        # mQ layout is (batch_size, seqlen_q, num_heads, head_dim)
        # sQ layout is (m_block_size, head_dim_padded), m_block_size is a divide for seqlen_q
        # swizzle(3, 3, 3), 8 bf16 (4 banks) as an element, 8 rows, 8 cols
        sQ_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size), stride=(smem_k_block_size, 1)),
        )
        cute.printf("sQ_layout_atom: {}\n", sQ_layout_atom)
        sQ_layout = cute.tile_to_shape(
            sQ_layout_atom,
            (self._m_block_size, self._head_dim_padded),
            (0, 1),
        )
        cute.printf("sQ_layout: {}\n", sQ_layout)

        sKV_layout_atom = sQ_layout_atom
        sKV_layout = cute.tile_to_shape(
            sKV_layout_atom,
            (self._n_block_size, self._head_dim_padded),
            (0, 1),
        )

        sO_layout = sQ_layout

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]
            sO: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sO_layout)], 1024
            ]

        # 16 bytes, 4 banks
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self._dtype.width
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        tQKV_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        tQKV_layout = cute.make_layout(
            (self._num_threads // tQKV_shape_dim_1, tQKV_shape_dim_1),
            stride=(tQKV_shape_dim_1, 1),
        )
        tO_layout = tQKV_layout
        vQKV_layout = cute.make_layout((1, async_copy_elems))
        vO_layout = vQKV_layout

        gmem_tiled_copy_QKV = cute.make_tiled_copy_tv(
            atom_async_copy, tQKV_layout, vQKV_layout
        )
        gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy, tO_layout, vO_layout
        )

        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (self._num_threads // 32, 1, 1),
            permutation_mnk=(self._num_threads // 32 * 16, 16, 16),
        )

        grid_dim = (
            cute.ceil_div(mQ.shape[1], self._m_block_size),
            cute.size(mQ.shape[0]),
            cute.size(mQ.shape[2]),
        )
        LOG2_E = 1.4426950408889634074
        softmax_scale_log2 = softmax_scale * LOG2_E
        # self.kernel(
        #     mQ,
        #     mK,
        #     mV,
        #     mO,
        #     softmax_scale_log2,
        #     sQ_layout,
        #     sKV_layout,
        #     sO_layout,
        #     gmem_tiled_copy_QKV,
        #     gmem_tiled_copy_O,
        #     tiled_mma,
        #     SharedStorage,
        # ).launch(
        #     grid=grid_dim,
        #     block=[self._num_threads, 1, 1],
        #     stream=stream,
        # )


if __name__ == "__main__":
    batch_size = 4
    dtype = cutlass.BFloat16
    seqlen_q = 8192
    seqlen_k = 8192
    num_heads = 16
    head_dim = 128
    softmax_scale = 0.5
    m_block_size = 128
    n_block_size = 64
    num_threads = 128
    is_causal = True

    print("Running FlashAttentionV2Forward test with:")
    print(f"  dtype: {dtype}")
    print(f"  batch_size: {batch_size}")
    print(f"  seqlen_q: {seqlen_q}")
    print(f"  seqlen_k: {seqlen_k}")
    print(f"  num_head: {num_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  softmax_scale: {softmax_scale}")
    print(f"  m_block_size: {m_block_size}")
    print(f"  n_block_size: {n_block_size}")
    print(f"  num_threads: {num_threads}")
    print(f"  is_causal: {is_causal}")

    def create_tensor(
        batch_size: int,
        seqlen: int,
        num_heads: int,
        head_dim: int,
        dtype: Type[cutlass.Numeric],
    ) -> cute.Tensor:
        shape = (batch_size, seqlen, num_heads, head_dim)
        torch_tensor = (
            torch.empty(*shape, dtype=torch.int32)
            .random_(-2, 2)
            .to(dtype=cutlass_torch.dtype(dtype))
            .cuda()
        )
        cute_tensor = (
            from_dlpack(torch_tensor, assumed_align=16)
            .mark_layout_dynamic(leading_dim=3)
            .mark_compact_shape_dynamic(
                mode=3,
                stride_order=torch_tensor.dim_order(),
                divisibility=(128 // dtype.width),
            )
        )
        return cute_tensor, torch_tensor

    q, q_torch = create_tensor(batch_size, seqlen_q, num_heads, head_dim, dtype)
    k, k_torch = create_tensor(batch_size, seqlen_k, num_heads, head_dim, dtype)
    v, v_torch = create_tensor(batch_size, seqlen_k, num_heads, head_dim, dtype)
    o, o_torch = create_tensor(batch_size, seqlen_q, num_heads, head_dim, dtype)

    fa2_fwd = FlashAttentionV2Forward(
        head_dim,
        m_block_size,
        n_block_size,
        num_threads,
        is_causal,
    )

    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)
    compiled_fa2_fwd = cute.compile(fa2_fwd, q, k, v, o, softmax_scale, current_stream)

    compiled_fa2_fwd(q, k, v, o, softmax_scale, current_stream)
    torch.cuda.synchronize()
    q_ref = q_torch.permute(0, 2, 1, 3)
    k_ref = k_torch.permute(0, 2, 1, 3)
    v_ref = v_torch.permute(0, 2, 1, 3)
    torch.backends.cuda.enable_flash_sdp(enabled=True)
    ref_o = torch.nn.functional.scaled_dot_product_attention(
        q_ref, k_ref, v_ref, scale=softmax_scale, is_causal=is_causal
    ).permute(0, 2, 1, 3)

    torch.testing.assert_close(o_torch.cpu(), ref_o.cpu(), atol=1e-02, rtol=1e-04)
    print("Results verified successfully!")

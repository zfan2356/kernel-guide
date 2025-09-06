from contextlib import contextmanager

import torch
import triton
import triton.language as tl
import triton.profiler as proton
from triton.tools.tensor_descriptor import TensorDescriptor


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def supports_ws():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 10


def _gemm_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K, WS = args["M"], args["N"], args["K"], args.get("WARP_SPECIALIZE", False)
    ws_str = "_ws" if WS else ""
    ret["name"] = f"{kernel.name}{ws_str} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


# tma tensor map
HAS_TENSOR_DESC = supports_tma() and hasattr(tl, "make_tensor_descriptor")

HAS_HOST_TENSOR_DESC = supports_tma() and hasattr(
    triton.tools.tensor_descriptor, "TensorDescriptor"
)

HAS_WARP_SPECIALIZE = supports_ws() and HAS_TENSOR_DESC


def gemm_tma_persistent_get_configs():
    def gemm_tma_set_block_size_hook(nargs):
        EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", False)
        BLOCK_M = nargs["BLOCK_M"]
        BLOCK_N = nargs["BLOCK_N"]
        BLOCK_K = nargs["BLOCK_K"]
        nargs["a_desc"].block_shape = [BLOCK_M, BLOCK_K]
        nargs["b_desc"].block_shape = [BLOCK_N, BLOCK_K]
        if EPILOGUE_SUBTILE:
            nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N // 2]
        else:
            nargs["c_desc"].blok_shape = [BLOCK_M, BLOCK_N]

    return [
        triton.Config(
            {
                "BLOCK_M": BM,
                "BLOCK_N": BN,
                "BLOCK_K": BK,
                "GROUP_SIZE_M": 8,
                "EPILOGUE_SUBTILE": SUBTILE,
            },
            num_warps=w,
            num_stages=s,
            pre_hook=gemm_tma_set_block_size_hook,
        )
        for BM in [128]
        for BN in [128, 256]
        for BK in [64, 128]
        for s in [2, 3, 4]
        for w in [4, 8]
        for SUBTILE in [True, False]
    ]


@triton.jit
def _compute_tile(tile_id, num_tiles_in_group, num_m_tiles, GROUP_SIZE):
    group_id = tile_id // num_tiles_in_group
    first_tid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.autotune(
    configs=gemm_tma_persistent_get_configs(),
    key=["M", "N", "K", "WARP_SPECIALIZE"],
)
@triton.jit(launch_metadata=_gemm_launch_metadata)
def _gemm_kernel(
    a_ptr: tl.tensor,
    b_ptr: tl.tensor,
    c_ptr: tl.tensor,
    M: int,
    N: int,
    K: int,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    """
    a_ptr: [M, K], dtype=torch.bfloat16
    b_ptr: [K, N], dtype=torch.bfloat16
    c_ptr: [M, N], dtype=torch.float32
    """
    num_m_tiles = tl.cdiv(M, BLOCK_M)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    num_tiles = num_m_tiles * num_n_tiles

    num_tiles_in_group = GROUP_SIZE * num_n_tiles

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[K, N],
        strides=[N, 1],
        block_shape=[BLOCK_K, BLOCK_N],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N if not EPILOGUE_SUBTILE else BLOCK_N // 2],
    )

    for tile_id in tl.range(
        tl.program_id(0),
        num_tiles,
        NUM_SMS,
        flatten=True,
        warp_specialize=WARP_SPECIALIZE,
    ):
        tile_m, tile_n = _compute_tile(
            tile_id, num_tiles_in_group, num_m_tiles, GROUP_SIZE
        )
        block_m = tile_m * BLOCK_M
        block_n = tile_n * BLOCK_N

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for block_k in range(0, K, BLOCK_K):
            a = a_desc.load([block_m, block_k])
            b = b_desc.load([block_k, block_n])
            accumulator = tl.dot(a, b, accumulator)

        c_desc.store([block_m, block_n], accumulator.to(tl.float32))


def gemm_nt(a: torch.Tensor, b: torch.Tensor, num_sms: int):
    assert a.ndim == 2 and b.ndim == 2, "a and b must be 2D tensors"
    assert a.shape[1] == b.shape[0], "a.shape[1] must be equal to b.shape[0]"
    assert a.dtype == b.dtype, "a and b must have the same dtype"

    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)

    with torch.cuda.device(a.device):
        _gemm_kernel[num_sms,](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
        )
    return c

import torch
import triton
import triton.language as tl


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


def native_gemm_get_configs(pre_hook=None):
    return [
        triton.Config(
            {
                "BLOCK_M": BLOCK_M,
                "BLOCK_N": BLOCK_N,
                "BLOCK_K": BLOCK_K,
                "GROUP_SIZE": 8,
            },
            num_stages=s,
            num_warps=w,
            pre_hook=pre_hook,
        )
        for BLOCK_M in [128]
        for BLOCK_N in [128, 256]
        for BLOCK_K in [64, 128]
        for s in [3, 4]
        for w in [4, 8]
    ]


@triton.autotune(
    configs=native_gemm_get_configs(),
    key=["M", "N", "K"],
)
@triton.jit(launch_metadata=_gemm_launch_metadata)
def _native_gemm_kernel(
    a_ptr: tl.tensor,
    b_ptr: tl.tensor,
    c_ptr: tl.tensor,
    M: int,
    N: int,
    K: int,
    stride_am: int,
    stride_ak: int,
    stride_bk: int,
    stride_bn: int,
    stride_cm: int,
    stride_cn: int,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """
    a_ptr: [M, K]
    b_ptr: [K, N]
    c_ptr: [M, N]
    """
    pid = tl.program_id(axis=0)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    num_n_blocks = tl.cdiv(N, BLOCK_N)

    # re-order for l2 cache
    num_blocks_in_group = GROUP_SIZE * num_n_blocks
    group_id = pid // num_blocks_in_group
    first_block = group_id * GROUP_SIZE
    group_size_m = min(num_m_blocks - first_block, GROUP_SIZE)

    block_m = first_block + (pid % group_size_m)
    block_n = (pid % num_blocks_in_group) // group_size_m
    start_m = block_m * BLOCK_M
    start_n = block_n * BLOCK_N

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_m = tl.where(
        offs_m < M, offs_m, 0
    )  # returns a tensor for elements from either offs_am or 0 depending on the condition(offs_am < M)
    offs_n = tl.where(offs_n < N, offs_n, 0)

    # Let the compiler know that the offs is contiguous and multiple of BLOCK_M.
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if c_ptr.dtype.element_ty == tl.float8e4nv:
        c = accumulator.to(tl.float8e4nv)
    else:
        c = accumulator.to(tl.float16)

    tl.store(
        c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :],
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def native_gemm_nt(a: torch.Tensor, b: torch.Tensor, num_sms: int):
    assert a.ndim == 2 and b.ndim == 2, "a and b must be 2D tensors"
    assert a.shape[1] == b.shape[0], "a.shape[1] must be equal to b.shape[0]"
    assert a.dtype == b.dtype, "a and b must have the same dtype"

    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    with torch.cuda.device(a.device):
        _native_gemm_kernel[num_sms,](
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

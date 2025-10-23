import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

M, N, K, L = 2, 32, 64, 128
tile_m, tile_n = 32, 16

tensor = torch.randn((M, N, K, L), device="cpu", dtype=torch.float32)
_tensor = from_dlpack(tensor, assumed_align=16)


@cute.jit
def _rank(tensor: cute.Tensor):
    print("===== Rank Example ====")
    layout = cute.make_layout(shape=(1, 2, (3, 5)), stride=(1, 1, (2, 6)))
    print(f"layout's rank is {cute.rank(layout)}")  # layout's rank is 3

    print(f"tensor's rank is {cute.rank(tensor)}")  # tensor's rank is 4


@cute.jit
def _group_modes(
    tensor: cute.Tensor,
):
    print("===== Group Modes Example ====")
    rank = cute.rank(tensor)
    # before group modes, tensor's layout is (2,32,64,128):(262144,8192,128,1), mode[0] is (2):(262144)
    print(
        f"before group modes, tensor's layout is {tensor.layout}, mode[0] is {cute.select(tensor.layout, mode=[0])}"
    )
    # before group modes, tensor[None, 0, 0, 0]'s layout is (2):(262144)
    print(
        f"before group modes, tensor[None, 0, 0, 0]'s layout is {tensor[None, 0, 0, 0].layout}"
    )

    grouped_tensor = cute.group_modes(tensor, 0, rank - 1)
    # after group modes, tensor's layout is ((2,32,64),128):((262144,8192,128),1), mode[0] is ((2,32,64)):((262144,8192,128))
    print(
        f"after group modes, tensor's layout is {grouped_tensor.layout}, mode[0] is {cute.select(grouped_tensor.layout, mode=[0])}"
    )
    # after group modes, tensor[None, 0]'s layout is ((2,32,64)):((262144,8192,128))
    print(
        f"after group modes, tensor[None, 0]'s layout is {grouped_tensor[None, 0].layout}"
    )


if __name__ == "__main__":
    _rank(_tensor)
    _group_modes(_tensor)

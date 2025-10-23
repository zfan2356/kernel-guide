import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass.cute.runtime import from_dlpack

num_threads = 256
_bM, _bN, _bK = 128, 128, 8


@cute.jit
def tiled_mma(mC: cute.Tensor):
    atoms_layout = cute.make_layout((num_threads // 16, 16, 1), stride=(16, 1, 0))
    if cutlass.const_expr(
        utils.LayoutEnum.from_tensor(mC) == utils.LayoutEnum.COL_MAJOR
    ):
        atoms_layout = cute.make_layout((16, num_threads // 16, 1), stride=(1, 16, 0))
    op = cute.nvgpu.MmaUniversalOp(cutlass.Float32)
    permutation_tiler_M = cute.make_layout((atoms_layout.shape[0], 4), stride=(4, 1))
    permutation_tiler_N = cute.make_layout((atoms_layout.shape[1], 4), stride=(4, 1))
    tiled_mma = cute.make_tiled_mma(
        op,
        atoms_layout,
        permutation_mnk=(permutation_tiler_M, permutation_tiler_N, None),
    )

    print(f"Atoms layout: {atoms_layout}")
    print("===== A Tile Layout ====")
    print(f"TiledMMA TV-layout-A: {tiled_mma.tv_layout_A_tiled}")
    print("===== C Tile Layout ====")
    print(f"TiledMMA TV-layout-C: {tiled_mma.tv_layout_C_tiled}")


if __name__ == "__main__":
    c = torch.randn(1024, 1024, device="cuda", dtype=torch.float)
    _c = from_dlpack(c, assumed_align=16)

    tiled_mma(_c)

# output
# Atoms layout: (16,16,1):(16,1,0)
# TiledMMA TV-layout-A: ((16,16),(1,(4,1))):((0,4),(0,(1,0)))
# TiledMMA TV-layout-C: ((16,16),(1,(4,4))):((256,4),(0,(1,64)))

# without Permutation:
# Atoms layout: (16,16,1):(16,1,0)
# TiledMMA TV-layout-A: ((16,16),(1,(1,1))):((0,1),(0,(0,0)))
# TiledMMA TV-layout-C: ((16,16),(1,(1,1))):((16,1),(0,(0,0)))

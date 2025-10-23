"""
Op: like
    - cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL) it is a cp.async op
    - cpasync.CopyBulkTensorTileG2SOp() it is a tma op

======= Next Step ========
Copy Atom:
    - cute.make_copy_atom(Op): make a copy atom
    - cpasync.make_tiled_tma_atom(Op): make a tiled tma copy atom

======= Next Step ========
Tiled Copy:
    - cpasync.tma_partition(
        atom,
        cta_coord,
        cta_layout,
        cute.group_modes(smem_tensor, 0, group_rank_smem),
        cute.group_modes(gmem_tensor, 0, group_rank_gmem),
    ) seems that it can get a tma tiled copy, and returns thr partition if S and D

    - cute.make_tiled_copy(
        atom=atom,
        layout_tv=tv_layout,
        tiler_mn=block_tiler,
    ) it can get a tiled copy, we will use get_slice(thread_idx) to get thr partiton, then get
        partition_S, and partition_D

    so Tiled Copy's destination is get the partition of thread

======= Next Step ========
Do Actually Copy Operation:
    - cute.copy(atom, partition_S, fragment) do actually copy operation
        can use cp.async

    - cute.copy(atom, partition_S, partition_D) do actually copy operation with tma
        can pass S and D, through Tiled Copy's Step

"""

if __name__ == "__main__":
    pass

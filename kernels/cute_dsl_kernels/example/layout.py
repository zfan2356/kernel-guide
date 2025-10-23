import cutlass.cute as cute
from cute_viz import render_layout_svg

"""
Layout:

Some important functions:
    cute.coalesce(layout): can simplify the layout, and make sure layout(i) == result(i) for all i in cute.size(layout)

    cute.compose(inner, outer, offset): suspect the result is R, inner is A, outer is B, so it satisfies R(i) = B(A(i)) + offset
        - so, R.shape == B.shape
        - it has a useful way, cute.composition(layout, tiler) to make a tiler to layout
        - see layout_composition_before.svg and layout_composition.svg





Reference:
    https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/notebooks/cute_layout_algebra.ipynb
"""


@cute.jit
def _inner_outer_layout():
    print("===== Inner Outer Layout Example ====")
    # layout = cute.make_layout(shape=(1, 2, (3, 5)), stride=(1, 1, (2, 6)))
    l1 = cute.make_layout(shape=(2, 4), stride=(1, 2))
    l2 = cute.make_layout(shape=(3, 5), stride=(5, 1))
    layout = cute.make_composed_layout(inner=l1, outer=l2, offset=2)

    print(f"layout.inner is {layout.inner}")  # layout.inner is (1, 2):(1, 1)
    print(f"layout.outer is {layout.outer}")  # layout.outer is (3, 5):(5, 1)

    print(
        f"composed layout is {layout}"
    )  # composed layout is (1,2):(1,1) o 2 o (3,5):(5,1), means that: inner o offset o outer

    layout2 = cute.composition(l1, l2)
    print(f"layout2 is {layout2}")

    render_layout_svg(layout, "layout_composed.svg")

    layout = cute.make_layout(
        shape=(12, (4, 8)),
        stride=(59, (13, 1)),
    )
    # layout is ((2,2),(2,4)):((1,4),(2,8))
    print(f"layout is {layout}, coalesced layout is {cute.coalesce(layout)}")
    render_layout_svg(layout, "layout_composition_before.svg")

    tiler = (3, 8)
    layout = cute.composition(layout, tiler)
    # tiler is (3, 8), by mode composed layout is (3,(4,2)):(59,(13,1))
    print(f"tiler is {tiler}, by mode composed layout is {layout}")

    render_layout_svg(layout, "layout_composition.svg")


@cute.jit
def _coalesce():
    print("===== Coalesce Example ====")
    layout = cute.make_layout(shape=((2, 2), (2, 4)), stride=((1, 4), (2, 8)))
    # layout is ((2,2),(2,4)):((1,4),(2,8))
    print(f"layout is {layout}")

    # coalesced layout is (2,2,2,4):(1,4,2,8)
    print(f"coalesced layout is {cute.coalesce(layout)}")

    render_layout_svg(layout, "layout_coalesce.svg")


@cute.jit
def _divide():
    print("===== Divide Example ====")
    layout = cute.make_layout((4, 2, 3), stride=(2, 1, 8))
    # layout is (4,2,3):(2,1,8)
    print(f"layout is {layout}")

    tiler = cute.make_layout(4, stride=2)
    result = cute.logical_divide(layout, tiler=tiler)
    print(f"divide layout is {result}")

    render_layout_svg(result, "layout_divide.svg")


if __name__ == "__main__":
    _inner_outer_layout()
    _coalesce()

    _divide()

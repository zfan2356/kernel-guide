import cutlass
import cutlass.cute as cute


@cute.jit
def layout_function_example():
    """
    Layout function in cutlass
    """
    S = (2, 4)
    D = (2, 2)
    L = cute.make_layout(shape=S, stride=D)

    for i in cutlass.range_constexpr(cute.size(S)):
        cute.printf("fL({}) = {}", i, L(i))


if __name__ == "__main__":
    layout_function_example()

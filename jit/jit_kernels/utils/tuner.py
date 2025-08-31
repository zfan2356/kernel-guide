import copy
import functools
import os
from typing import Any, Dict

import torch
from jit_version.jit import Runtime, build, cpp_format, generate


@functools.cache
def _jit_debug():
    return os.getenv("JIT_DEBUG", None) is not None


@functools.cache
def _jit_print_autotune():
    return os.getenv("PRINT_AUTOTUNE", None) is not None


class JITTuner:
    def __init__(self) -> None:
        self.tuned = {}

    def compile_and_tune(
        self,
        name: str,
        keys: Dict[str, Any],
        space: tuple,
        includes: tuple,
        arg_defs: tuple,
        template: str,
        args: tuple,
    ) -> Runtime:
        # NOTES: we always assume the space and template will not change
        # We also assume the GPU device will not be changed
        # NOTES: the function must have no accumulated side effects
        keys = {k: keys[k] for k in sorted(keys.keys())}
        signature = (name, f"{keys}")
        if signature in self.tuned:
            if _jit_debug():
                print(f"Using cached JIT kernel {name} with keys {keys}")
            return self.tuned[signature]

        if _jit_debug():
            print(f"Auto-tuning JIT kernel {name} with keys {keys}")

        assert signature not in self.tuned
        assert args is not None
        space = (dict(),) if len(space) == 0 else space

        kernels = []
        for tuned_keys in space:
            assert isinstance(tuned_keys, dict)
            full_keys = copy.deepcopy(keys)
            full_keys.update(tuned_keys)
            code = generate(includes, arg_defs, cpp_format(template, full_keys))

            # Illegal build must raise errors
            kernels.append((build(name, arg_defs, code), tuned_keys))

        best_runtime, best_time, best_keys = None, None, None
        for runtime, tuned_keys in kernels:
            if len(space) > 1:
                # Check kernel validity
                return_code = runtime(*args)
                if return_code != 0:
                    # Pass illegal kernels, e.g. insufficient shared memory capacity
                    if _jit_debug():
                        print(
                            f"Illegal JIT kernel {name} with keys {keys} and tuned keys {tuned_keys}: error code {return_code}"
                        )
                    continue

                # Measure performance with L2 flush and a large GEMM kernel before to reduce overhead between kernels
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda").zero_()
                torch.randn(
                    (8192, 8192), dtype=torch.float, device="cuda"
                ) @ torch.randn((8192, 8192), dtype=torch.float, device="cuda")
                start_event.record()
                for i in range(20):
                    assert runtime(*args) == 0
                end_event.record()
                end_event.synchronize()
                elapsed_time = start_event.elapsed_time(end_event)
            else:
                elapsed_time = 0

            # Compare if better
            if best_time is None or elapsed_time < best_time:
                best_runtime, best_time, best_keys = runtime, elapsed_time, tuned_keys
            if _jit_debug():
                print(
                    f"Tuned JIT kernel {name} with keys {keys} and tuned keys {tuned_keys} has time {elapsed_time}"
                )

        assert (
            best_runtime is not None
        ), f"Failed to tune JIT kernel {name} with keys {keys}"

        # Cache the best runtime and return
        if _jit_debug() or _jit_print_autotune():
            print(
                f"Best JIT kernel {name} with keys {keys} has tuned keys {best_keys} and time {best_time}"
            )

        self.tuned[signature] = best_runtime
        return best_runtime


jit_tuner = JITTuner()

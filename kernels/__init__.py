import os
import subprocess

from kernels_cpp import cp_async_test, hello_world, init


# Initialize CPP modules
def _find_cuda_home() -> str:
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is None:
        try:
            with open(os.devnull, "w") as devnull:
                nvcc = (
                    subprocess.check_output(["which", "nvcc"], stderr=devnull)
                    .decode()
                    .rstrip("\r\n")
                )
                cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            cuda_home = "/usr/local/cuda"
            if not os.path.exists(cuda_home):
                cuda_home = None
    assert cuda_home is not None
    return cuda_home


init(
    os.path.dirname(os.path.abspath(__file__)),  # Library root directory path
    _find_cuda_home(),  # CUDA home
)

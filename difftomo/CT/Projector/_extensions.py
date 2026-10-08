from pathlib import Path
import os
import importlib.abc
from torch.utils.cpp_extension import load


os.environ.setdefault("CC", "/usr/bin/gcc-12")
os.environ.setdefault("CXX", "/usr/bin/g++-12")

_PROJECTOR_DIR = Path(__file__).parent

difftomo_cuda = load(
    name="difftomo_cuda",
    sources=[
        str(_PROJECTOR_DIR / "bindings.cpp"),
        str(_PROJECTOR_DIR / "SiddonParallelBeam.cu"),
    ],
    verbose=True,
)

forward_siddon = difftomo_cuda.forward_siddon
backward_siddon = difftomo_cuda.backward_siddon

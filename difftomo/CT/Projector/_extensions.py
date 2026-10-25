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
        str(_PROJECTOR_DIR / "Cpp_files" / "bindings_cuda.cpp"),
        str(_PROJECTOR_DIR / "Cuda_files" / "SiddonAlgorithm.cu"),
    ],
    verbose=True,
)

difftomo_cpu = load(
    name="difftomo_cpu",
    sources=[
        str(_PROJECTOR_DIR / "Cpp_files" / "bindings_cpu.cpp"),
        str(_PROJECTOR_DIR / "Cpp_files" / "SiddonAlgorithm.cpp"),
        str(_PROJECTOR_DIR / "Cpp_files" / "ARTAlgo.cpp"),
    ],
    extra_include_paths=[str(_PROJECTOR_DIR)],
    verbose=True,
)

forward_siddon = difftomo_cuda.forward_siddon
backward_siddon = difftomo_cuda.backward_siddon

forward_siddon_cpu = difftomo_cpu.forward_siddon_cpu
backward_siddon_cpu = difftomo_cpu.backward_siddon_cpu
ARTReconstruction = difftomo_cpu.ARTRecon

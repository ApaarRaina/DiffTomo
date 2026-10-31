from pathlib import Path
import sys
import torch
import os
import shutil
from torch.utils.cpp_extension import load, CUDA_HOME


_PROJECTOR_DIR = Path(__file__).parent

_CPP_DIR = _PROJECTOR_DIR / "Cpp_files"
_CUDA_DIR = _PROJECTOR_DIR / "Cuda_files"

_cpu_extension = None
_cuda_extension = None

if shutil.which("gcc-12") and shutil.which("g++-12"):
    os.environ.setdefault("CC", shutil.which("gcc-12"))
    os.environ.setdefault("CXX", shutil.which("g++-12"))


def _load_cpu():
    global _cpu_extension

    if _cpu_extension is None:
        _cpu_extension = load(
            name="difftomo_cpu",
            sources=[
                str(_CPP_DIR / "bindings_cpu.cpp"),
                str(_CPP_DIR / "SiddonAlgorithm.cpp"),
                str(_CPP_DIR / "ARTAlgo.cpp"),
            ],
            extra_include_paths=[str(_PROJECTOR_DIR)],
            extra_cflags=["-fopenmp"] if sys.platform.startswith("linux") else [],
            extra_ldflags=["-fopenmp"] if sys.platform.startswith("linux") else [],
            verbose=True,
        )

    return _cpu_extension


def _load_cuda():
    global _cuda_extension

    if _cuda_extension is not None:
        return _cuda_extension

    if CUDA_HOME is None:
        raise RuntimeError(
            "DiffTomo CUDA support requires the NVIDIA CUDA Toolkit. "
            "Install a compatible CUDA development toolkit or use device='cpu'."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Check your NVIDIA driver and PyTorch CUDA "
            "installation, or use device='cpu'."
        )

    _cuda_extension = load(
        name="difftomo_cuda",
        sources=[
            str(_CPP_DIR / "bindings_cuda.cpp"),
            str(_CUDA_DIR / "SiddonAlgorithm.cu"),
        ],
        extra_include_paths=[str(_PROJECTOR_DIR)],
        verbose=True,
    )

    return _cuda_extension


# CUDA functions: compiled only when first called.

def forward_siddon(*args, **kwargs):
    return _load_cuda().forward_siddon(*args, **kwargs)


def backward_siddon(*args, **kwargs):
    return _load_cuda().backward_siddon(*args, **kwargs)


# CPU functions: compiled only when first called.

def forward_siddon_cpu(*args, **kwargs):
    return _load_cpu().forward_siddon_cpu(*args, **kwargs)


def backward_siddon_cpu(*args, **kwargs):
    return _load_cpu().backward_siddon_cpu(*args, **kwargs)


def ARTReconstruction(*args, **kwargs):
    return _load_cpu().ARTRecon(*args, **kwargs)
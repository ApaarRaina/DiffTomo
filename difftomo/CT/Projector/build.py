from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="difftomo_cuda",
    ext_modules=[
        CUDAExtension(
            name="difftomo_cuda",
            sources=[
                "bindings.cpp",
                "Siddon_algo.cu"
            ],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)
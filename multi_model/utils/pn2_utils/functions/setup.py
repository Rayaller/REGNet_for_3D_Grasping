import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

DEFAULT_TORCH_CUDA_ARCH_LIST = "6.0;6.1;7.0;7.5;8.0;8.6+PTX"

if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
    os.environ["TORCH_CUDA_ARCH_LIST"] = DEFAULT_TORCH_CUDA_ARCH_LIST
    print(f"[dgcnn_ext setup] TORCH_CUDA_ARCH_LIST={DEFAULT_TORCH_CUDA_ARCH_LIST}")

extra_compile_args = {'cxx': ['-g'],
                      'nvcc': ['-O2']}

setup(
    name='dgcnn_ext',
    ext_modules=[
        CUDAExtension(
            name='dgcnn_ext',
            sources=[
                'csrc/main.cpp',
                'csrc/gather_knn_kernel.cu',
            ],
            extra_compile_args=extra_compile_args
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='emd_cuda',
    ext_modules=[
        CUDAExtension('emd_cuda', [
            'emd_cuda.cpp',
            'emd_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
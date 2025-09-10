from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = "0.0.1"

# Define the C++ extension modules
ext_modules = [
    CUDAExtension('cusolve', [
        'roll_call.cu',
        'roll_call_binding.cpp',
    ])
]

setup(
    name="cusolve",
    version=__version__,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension}
)
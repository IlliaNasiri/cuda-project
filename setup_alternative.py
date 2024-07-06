from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import os
import shutil
import numpy as np
import sysconfig

current_path = os.getcwd()
cuda_dynamic_lib_directory = sysconfig.get_config_var('LIBDIR')

if not os.path.exists(cuda_dynamic_lib_directory):
    os.makedirs(cuda_dynamic_lib_directory)

shutil.copy(os.path.join(current_path, "extensions/src/libgpumat.so"), cuda_dynamic_lib_directory )
shutil.copy(os.path.join(current_path, "extensions/src/libmatmul.so"), cuda_dynamic_lib_directory )

ext_modules = [
    Extension("gpumat",
              sources=["GpuMatWrapper.pyx"],
              include_dirs=['extensions/src/', np.get_include()],
              libraries=["gpumat", "matmul"])
]


setup(name="add2_kernel",
      packages=find_packages(),
      ext_modules=cythonize(ext_modules))

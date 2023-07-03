from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='kfloat_cpp',
      ext_modules=[cpp_extension.CppExtension('kfloat_cpp', ['binding.cpp', 'util.cpp', 'kmeans.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
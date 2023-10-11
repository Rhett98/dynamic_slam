import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension


ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]
sources = glob.glob('*.cpp')


setup(
    name='knn',
    version='1.0',
    author='rheet98',
    author_email='37391367@qq.com',
    description='knn',
    long_description='knn',
    ext_modules=[
        CppExtension(
            name='knn',
            sources=sources,
            include_dirs=include_dirs,
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
"""
ZION Yescrypt C Extension Module
High-performance Yescrypt implementation in C for maximum mining speed
Compile with: python setup.py build_ext --inplace
"""

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension
import os

# C extension for optimized Yescrypt
yescrypt_module = Extension(
    'yescrypt_fast',
    sources=[
        'yescrypt_fast.c',
        'salsa20.c'  # Will be created separately
    ],
    include_dirs=['.'],
    extra_compile_args=['/O2'],  # Windows MSVC optimization
    extra_link_args=[]
)

setup(
    name='ZION Yescrypt Fast',
    version='2.7.1',
    description='High-performance Yescrypt implementation for ZION mining',
    ext_modules=[yescrypt_module],
    author='ZION Development Team',
    author_email='dev@zion.network'
)
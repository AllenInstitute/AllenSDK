from setuptools import setup, find_packages

setup(
    version = '0.1',
    namespace_packages = ['allen_wrench'],
    name = 'allen_wrench.model',
    packages = find_packages(),
    description = 'model libraries for the allen_wrench.',
    install_requires = ['setuptools', 'h5py', 'allen_wrench.core'],
)

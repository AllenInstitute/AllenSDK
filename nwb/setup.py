from setuptools import setup, find_packages

setup(
    version = '0.1',
    namespace_packages = ['allen_wrench'],
    name = 'allen_wrench.nwb',
    packages = find_packages(),
    description = '',
    install_requires = ['setuptools', 'h5py'],
)

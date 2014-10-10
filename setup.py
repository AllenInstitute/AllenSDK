from setuptools import setup, find_packages

setup(
    version = '0.2',
    name = 'allen_wrench',
    packages = find_packages(),
    description = 'core libraries for the allen_wrench.',
    install_requires = ['setuptools', 'h5py'],
)

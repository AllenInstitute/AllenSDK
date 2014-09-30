from setuptools import setup, find_packages

setup(
    name = "allen_wrench",
    version = "0.1",
    packages = find_packages(exclude=['allen_wrench.model','allen_wrench.vis']),
    install_requires = [ 'numpy', 'scipy', 'h5py' ],
    package_data = { '': ['*.py' ] }
)

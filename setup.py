import sys

from setuptools import setup, find_packages
from setuptools.command.install import install

packages = {
    'install_vis': [ 'allen_wrench.core', 'allen_wrench.vis' ],
    'install_model': [ 'allen_wrench.core', 'allen_wrench.model' ],
    'install': [ 'allen_wrench.core' ]
}

# this is probably a bad idea, as setuptools has command line args...
command_name = sys.argv[1]

setup(
    name = "allen_wrench",
    version = "0.1",
    packages = packages[command_name],
    install_requires = [ 'numpy', 'scipy', 'h5py' ],
    cmdclass = { k: install for k in packages.keys() },
    package_data = { '': ['*.py' ] }
)

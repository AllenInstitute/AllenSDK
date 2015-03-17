from setuptools import setup, find_packages
import os
import allen_wrench

# http://bugs.python.org/issue8876#msg208792
del os.link

def prepend_find_packages(*roots):
    ''' Recursively traverse nested packages under the root directories
    '''
    packages = []
    
    for root in roots:
        packages += [root]
        packages += [root + '.' + s for s in find_packages(root)]
        
    return packages

setup(
    version = allen_wrench.__version__,
    name = 'allen_wrench',
    author = 'David Feng',
    author_email = 'davidf@alleninstitute.org',
    packages = prepend_find_packages('allen_wrench'),
    package_data={'': ['*.hoc', '*.conf', '*.cfg', '*.md', '*.json', '*.dat', '*.xyz', '*.env', '*.sh', 'bps'] },
    description = 'core libraries for the allen_wrench.',
    requires = ['h5py',
                'argparse',
                'six'],
    tests_require=['nose>=1.2.1',
                   'coverage>=3.7.1'],
    setup_requires=['setuptools', 'sphinx'],
    url='http://',
    scripts=['allen_wrench/model/biophys_sim/scripts/bps'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: Apache Software License :: 2.0',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
        ])

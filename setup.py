from setuptools import setup, find_packages
import os
import allensdk

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
    version = allensdk.__version__,
    name = 'allensdk',
    author = 'David Feng',
    author_email = 'davidf@alleninstitute.org',
    packages = prepend_find_packages('allensdk'),
    package_data={'': ['*.conf', '*.cfg', '*.md', '*.json', '*.dat', '*.env', '*.sh', 'bps'] },
    description = 'core libraries for the allensdk.',
    requires = ['h5py',
                'argparse',
                'six'],
    tests_require=['nose>=1.2.1',
                   'coverage>=3.7.1'],
    setup_requires=['setuptools', 'sphinx'],
    url='http://',
    scripts=['allensdk/model/biophys_sim/scripts/bps'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
        ])

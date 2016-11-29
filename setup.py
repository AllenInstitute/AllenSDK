from setuptools import setup, find_packages
import os
import allensdk

# http://bugs.python.org/issue8876#msg208792
if hasattr(os, 'link'):
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
    package_data={'': ['*.conf', '*.cfg', '*.md', '*.json', '*.dat', '*.env', '*.sh', '*.txt', 'bps', 'Makefile', 'COPYING'] },
    description = 'core libraries for the allensdk.',
    install_requires = ['h5py>=2.2.1',
                        'matplotlib>=1.4.0',
                        'scipy>=0.14.0',
                        'pandas>=0.14.1',
                        'jinja2>=2.7.3',
                        'numpy>=1.8.2',
                        'six>=1.5.2',
                        'pynrrd >= 0.2.1',
                        'requests',
                        'requests-toolbelt'],
    tests_require=['pytest>=2.6.3',
                   'pytest-cov>=2.2.1',
                   'pytest-cover>=3.0.0',
                   'pytest-mock>=0.11.0',
                   'pytest-pep8>=1.0.6',
                   'coverage>=3.7.1',
                   'mock>=1.0.1'],
    setup_requires=['setuptools', 'sphinx', 'numpydoc'],
    url='https://github.com/AllenInstitute/AllenSDK/tree/v%s' % (allensdk.__version__),
    download_url = 'https://github.com/AllenInstitute/AllenSDK/tarball/v%s' % (allensdk.__version__),
    keywords = ['neuroscience', 'bioinformatics', 'scientific'  ],
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

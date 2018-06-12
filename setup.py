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

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

with open('test_requirements.txt', 'r') as f:
    test_required = f.read().splitlines()

setup(
    version = allensdk.__version__,
    name = 'allensdk',
    author = 'David Feng',
    author_email = 'davidf@alleninstitute.org',
    packages = prepend_find_packages('allensdk'),
    package_data={'': ['*.conf', '*.cfg', '*.md', '*.json', '*.dat', '*.env', '*.sh', '*.txt', 'bps', 'Makefile', 'LICENSE'] },
    description = 'core libraries for the allensdk.',
    install_requires = required,
    tests_require=test_required,
    setup_requires=['setuptools', 'sphinx', 'numpydoc', 'pytest-runner'],
    url='https://github.com/AllenInstitute/AllenSDK/tree/v%s' % (allensdk.__version__),
    download_url = 'https://github.com/AllenInstitute/AllenSDK/tarball/v%s' % (allensdk.__version__),
    keywords = ['neuroscience', 'bioinformatics', 'scientific'  ],
    scripts=['allensdk/model/biophys_sim/scripts/bps'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: Other/Proprietary License', # Allen Institute Software License
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6', 
        'Topic :: Scientific/Engineering :: Bio-Informatics'
        ])

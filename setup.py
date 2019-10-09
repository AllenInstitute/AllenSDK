from setuptools import setup, find_packages
import os
import allensdk
import re

def parse_requirements_line(line):
    split = line.split("#")
    return split[0].strip()

def parse_requirements_file(path):
    with open(path, 'r') as fil:
        return [parse_requirements_line(line) for line in fil.read().splitlines()]

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

required = parse_requirements_file('requirements.txt')

if os.environ.get('ALLENSDK_INTERNAL_REQUIREMENTS', 'false') == 'true':
    required.extend(parse_requirements_file('internal_requirements.txt'))

test_required = parse_requirements_file('test_requirements.txt')

setup(
    version = allensdk.__version__,
    name = 'allensdk',
    author = 'Allen Institute for Brain Science',
    author_email = 'instituteci@alleninstitute.org',
    packages = ['allensdk'],
    include_package_data = True,
    package_data={'allensdk': ['*.conf', '*.cfg', '*.md', '*.json', '*.dat', '*.env', '*.sh', '*.txt', 'bps', 'Makefile', 'LICENSE', '*.hoc', 'allensdk/brain_observatory/nwb/*.yaml'] },
    description = 'core libraries for the allensdk.',
    install_requires = required,
    tests_require=test_required,
    setup_requires=['setuptools', 'sphinx', 'numpydoc', 'pytest-runner'],
    url='https://github.com/AllenInstitute/AllenSDK/tree/v%s' % (allensdk.__version__),
    download_url = 'https://github.com/AllenInstitute/AllenSDK/tarball/v%s' % (allensdk.__version__),
    keywords = ['neuroscience', 'bioinformatics', 'scientific'  ],
    scripts=['allensdk/model/biophys_sim/scripts/bps'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: Other/Proprietary License', # Allen Institute Software License
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6', 
        'Programming Language :: Python :: 3.7', 
        'Topic :: Scientific/Engineering :: Bio-Informatics'
        ])

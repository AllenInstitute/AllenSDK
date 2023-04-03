import os

from setuptools import find_packages, setup

import allensdk


def parse_requirements_line(line):
    split = line.split("#")
    return split[0].strip()


def parse_requirements_file(path):
    with open(path, "r") as fil:
        return [
            parse_requirements_line(line) for line in fil.read().splitlines()
        ]


# http://bugs.python.org/issue8876#msg208792
if hasattr(os, "link"):
    del os.link


def prepend_find_packages(*roots):
    """Recursively traverse nested packages under the root directories"""
    packages = []

    for root in roots:
        packages += [root]
        packages += [root + "." + s for s in find_packages(root)]

    return packages


# Enviroment variable needed to properly install pytables on Windows
# for miniconda installs. See:
# https://github.com/PyTables/PyTables/issues/933#issuecomment-1072128725
if os.environ.get("CONDA_DLL_SEARCH_MODIFICATION_ENABLE", "0") == "0":
    os.environ["CONDA_DLL_SEARCH_MODIFICATION_ENABLE"] = "1"

required = parse_requirements_file("requirements.txt")

if os.environ.get("ALLENSDK_INTERNAL_REQUIREMENTS", "false") == "true":
    required.extend(parse_requirements_file("internal_requirements.txt"))

test_required = parse_requirements_file("test_requirements.txt")

setup(
    version=allensdk.__version__,
    name="allensdk",
    author="Allen Institute for Brain Science",
    author_email="instituteci@alleninstitute.org",
    packages=["allensdk"],
    include_package_data=True,
    package_data={
        "allensdk": [
            "*.conf",
            "*.cfg",
            "*.md",
            "*.json",
            "*.dat",
            "*.env",
            "*.sh",
            "*.txt",
            "bps",
            "Makefile",
            "LICENSE",
            "*.hoc",
            "allensdk/brain_observatory/nwb/*.yaml",
        ]
    },
    description="core libraries for the allensdk.",
    install_requires=required,
    tests_require=test_required,
    setup_requires=["setuptools", "sphinx", "numpydoc", "pytest-runner"],
    url="https://github.com/AllenInstitute/AllenSDK/tree/v%s"
    % (allensdk.__version__),
    download_url="https://github.com/AllenInstitute/AllenSDK/tarball/v%s"
    % (allensdk.__version__),
    keywords=["neuroscience", "bioinformatics", "scientific"],
    scripts=["allensdk/model/biophys_sim/scripts/bps"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License", # Allen Institute License
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)

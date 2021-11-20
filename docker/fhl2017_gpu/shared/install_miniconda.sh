#!/bin/bash
wget https://repo.continuum.io/miniconda/Miniconda3-3.10.1-Linux-x86_64.sh
/bin/bash Miniconda3-3.10.1-Linux-x86_64.sh -b -p /opt/conda
ln -s /opt/conda/bin/conda /usr/local/bin/conda

# DOCKER-VERSION 0.3.4
#
# docker build --tag alleninstitute/allensdk_ubuntu .
# docker run -it alleninstitute/allensdk_ubuntu /bin/bash
#
FROM ubuntu:trusty

MAINTAINER Tim Fliss <timf@alleninstitute.org>

RUN apt-get update
RUN apt-get upgrade -yq

RUN apt-get -yq install \
    pkg-config \
    libfreetype6-dev \
    build-essential \
    automake \
    libtool \
    bison \
    flex \
    python \
    python-pip \
    python-scipy \
    python-matplotlib \
    python-pandas \
    ipython \
    python-dev \
    libxext-dev \
    libncurses-dev \
    libhdf5-dev \
    wget \
    openmpi-bin \
    libopenmpi-dev \
    alien \
    unzip

# upgrade cython
RUN pip install \
    cython \
    sphinx \
    numpydoc \
    cycler

#neuron installation
RUN cd /root; wget http://www.neuron.yale.edu/ftp/neuron/versions/v7.4/nrn-7.4.x86_64.rpm
RUN cd /root; alien -i nrn-7.4.x86_64.rpm

RUN mkdir allensdk; \
    cd allensdk; \
    wget -O allensdk.zip http://stash.corp.alleninstitute.org/plugins/servlet/archive/projects/INF/repos/allensdk?at=refs%2Fheads%2Fdev; \
    unzip allensdk.zip; \
    rm allensdk.zip; \
    pip install .; \
    pip install -r test_requirements.txt; \
    pip install jupyter; \
    jupyter-notebook --generate-config

ENV PYTHONPATH=/usr/local/nrn/lib/python:$PYTHONPATH

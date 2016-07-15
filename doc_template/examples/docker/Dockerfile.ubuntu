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
    libfreetype6-dev

RUN apt-get -yq install \
    build-essential \
    automake \
    libtool \
    bison \
    flex
    
RUN apt-get -yq install \
    python \
    python-pip \
    python-scipy \
    python-matplotlib \
    ipython \
    python-dev
    
RUN apt-get -yq install \
    libxext-dev \
    libncurses-dev \
    libhdf5-dev \
    wget

RUN apt-get -yq install \
    openmpi-bin \
    libopenmpi-dev


# upgrade cython
RUN pip install cython
RUN pip install sphinx numpydoc

#neuron installation
RUN cd /root; wget http://www.neuron.yale.edu/ftp/neuron/versions/v7.4/nrn-7.4.x86_64.rpm
RUN cd /root; rpm -ivh nrn-7.4.x86_64.rpm

RUN pip install jupyter


RUN cd /root; wget https://github.com/AllenInstitute/AllenSDK/archive/release_0.12.zip
RUN cd /root; \
    unzip release_0.12.zip
RUN cd /root; \
    pip install ./AllenSDK-release_0.12; \
    cd AllenSDK-release_0.12; \
    pip install -r test_requirements.txt

ENV PYTHONPATH=/usr/local/nrn/lib/python:$PYTHONPATH
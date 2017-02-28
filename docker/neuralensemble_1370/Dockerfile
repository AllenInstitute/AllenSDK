#
# A Docker image for running allensdk with neuronal network simulations
#
# docker build --tag alleninstitute/neuralensemble_py3_1370 .
# docker run -it alleninstitute/allensdk:neuralensemble_1370 /bin/bash
# source ~/env/neurosci/bin/activate
#
FROM neuralensemble/simulation
MAINTAINER Tim Fliss <timf@alleninstitute.org>

USER root

# http://micro.stanford.edu/wiki/Install_HDF5
RUN apt-get update; \
    apt-get install -yq \
    pkg-config \
    libhdf5-dev \
    hdf5-tools \
    libfreetype6-dev \
    python-skimage \
    cython

ENV NRN=nrn-$NRN_VER NRN_RELEASE=7.4.rel-1370

WORKDIR /home/docker/packages

RUN wget https://www.neuron.yale.edu/ftp/neuron/versions/v$NRN_VER/v$NRN_RELEASE/nrn-$NRN_RELEASE.tar.gz; \
    tar xzf nrn-$NRN_RELEASE.tar.gz; \
    rm nrn-$NRN_RELEASE.tar.gz

WORKDIR $VENV/build

RUN mkdir $NRN; \
    cd $NRN; \
    $HOME/packages/$NRN/configure --with-paranrn --with-nrnpython=$VENV/bin/python --disable-rx3d --without-iv --prefix=$VENV; \
    make; make install; \
    cd src/nrnpython; $VENV/bin/python setup.py install; \
    cd $VENV/bin; rm $VENV/bin/nrnivmodl; ln -s ../x86_64/bin/nrnivmodl

USER docker
WORKDIR /home/docker

COPY requirements.txt test_requirements.txt ./

RUN $VENV/bin/pip install --upgrade pip; \
    $VENV/bin/pip install -r requirements.txt; \
    $VENV/bin/pip install -r test_requirements.txt

RUN cd $VENV/bin; ln -s ../x86_64/bin/nrniv

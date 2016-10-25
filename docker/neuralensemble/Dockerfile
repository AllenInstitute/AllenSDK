#
# A Docker image for running allensdk with neuronal network simulations
#
# docker build --tag alleninstitute/allensdk:neuralensemble .
# docker run -it alleninstitute/allensdk:neuralensemble /bin/bash
# source ~/env/neurosci/bin/activate
#
FROM neuralensemble/simulation
MAINTAINER Tim Fliss <timf@alleninstitute.org>

USER root

# http://micro.stanford.edu/wiki/Install_HDF5
RUN apt-get install -yq \
    pkg-config \
    libhdf5-dev \
    hdf5-tools \
    libfreetype6-dev

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

RUN apt-get -yq install unzip

USER docker
WORKDIR /home/docker

# TODO: replace w/ install from Github
RUN mkdir allensdk; \
    cd allensdk; \
    wget -O allensdk.zip https://github.com/AllenInstitute/AllenSDK/archive/v0.12.3.zip; \
    unzip allensdk.zip; \
    rm allensdk.zip; \
    $VENV/bin/pip install .; \
    $VENV/bin/pip install -r test_requirements.txt; \
    $VENV/bin/pip install jupyter; \
    $VENV/bin/jupyter-notebook --generate-config

RUN cd $VENV/bin; ln -s ../x86_64/bin/nrniv

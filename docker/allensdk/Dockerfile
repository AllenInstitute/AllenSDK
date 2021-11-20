# DOCKER-VERSION 2.2.0.5
#
# docker build --tag alleninstitute/allensdk .
# docker run -it alleninstitute/allensdk /bin/bash
#
FROM continuumio/anaconda3

LABEL maintainer="isaak.willett@alleninstitute.org"

WORKDIR /usr/src/app

# Add AllenSDK and NEURON libraries
RUN pip install allensdk neuron


# Install NEURON
RUN wget https://neuron.yale.edu/ftp/neuron/versions/v7.7/nrn-7.7.x86_64-linux.deb; \
	dpkg -i nrn-7.7.x86_64-linux.deb; \
	apt-get update; \
	apt-get install -y \
		make \
		gcc \
		build-essential \
		libncurses5-dev \
		libreadline-dev \
		libx11-dev;

# Setup example notebooks and python tests
RUN wget https://github.com/AllenInstitute/AllenSDK/archive/master.tar.gz; \
	tar xvzf master.tar.gz; \
	mv AllenSDK-master allensdk; \
	cd allensdk; \
	pip install -r test_requirements.txt; \

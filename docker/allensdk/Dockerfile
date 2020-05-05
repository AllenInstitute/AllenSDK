# DOCKER-VERSION 2.2.0.5
#
# docker build --tag alleninstitute/allensdk .
# docker run -it alleninstitute/allensdk /bin/bash
#
FROM continuumio/anaconda3

WORKDIR /root

# Add AllenSDK and NEURON libraries
RUN pip install -q allensdk neuron

# Install NEURON
RUN wget https://neuron.yale.edu/ftp/neuron/versions/v7.7/nrn-7.7.x86_64-linux.deb; \
	dpkg -i nrn-7.7.x86_64-linux.deb; \
	apt-get install -yq \
		make \
		gcc \
		build-essential \
		libncurses5-dev \
		libreadline-dev \
		libx11-dev;

# Setup example notebooks and python tests
RUN wget https://github.com/AllenInstitute/AllenSDK/archive/rc/1.7.1.tar.gz; \
	tar xvzf 1.7.1.tar.gz; \
	mv AllenSDK-rc-1.7.1 allensdk; \
	cd allensdk; \
	pip install -r test_requirements.txt; \

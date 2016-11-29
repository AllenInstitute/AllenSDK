# DOCKER-VERSION 1.11.1
#
# docker build --tag alleninstitute/allensdk:anaconda .
# docker run -it alleninstitute/allensdk:anaconda /bin/bash
#
FROM continuumio/anaconda:4.1.1

MAINTAINER Tim Fliss <timf@alleninstitute.org>

# neuron installation
WORKDIR /root
RUN wget http://www.neuron.yale.edu/ftp/neuron/versions/v7.4/v7.4.rel-1370/nrn-7.4.rel-1370.x86_64.deb; \
    dpkg -i nrn-7.4.rel-1370.x86_64.deb; \
    apt-get install -f; \
    dpkg -i nrn-7.4.rel-1370.x86_64.deb 

RUN apt-get install -yq \
    unzip \
    make

RUN wget -O allensdk.zip wget -O allensdk.zip https://github.com/AllenInstitute/AllenSDK/archive/v0.12.4.zip; \
    unzip allensdk.zip; \
    rm allensdk.zip; \
    mv AllenSDK-0.12.4 allensdk; \
    cd allensdk; \
    pip install .; \
    pip install -r test_requirements.txt

ENV PYTHONPATH /usr/local/nrn/lib/python

# DOCKER-VERSION 1.11.1
#
# docker build --tag alleninstitute/allensdk:anaconda_neuron_1370 .
# docker run -it alleninstitute/allensdk:anaconda_neuron_1370 /bin/bash
#
FROM continuumio/anaconda:4.1.1

MAINTAINER Tim Fliss <timf@alleninstitute.org>

# neuron installation
WORKDIR /root
RUN wget http://www.neuron.yale.edu/ftp/neuron/versions/v7.4/v7.4.rel-1370/nrn-7.4.rel-1370.x86_64.deb; \
    dpkg -i nrn-7.4.rel-1370.x86_64.deb; \
    apt-get install -f; \
    dpkg -i nrn-7.4.rel-1370.x86_64.deb; \
    apt-get install -yq unzip make 

COPY requirements.txt test_requirements.txt ./
RUN  pip install -r requirements.txt; \
     pip install -r test_requirements.txt

RUN apt-get -yq install libopenjp2-7

ENV PYTHONPATH /usr/local/nrn/lib/python
ENV PATH /usr/local/nrn/x86_64/bin:$PATH
ENV TEST_API_ENDPOINT http://api.brain-map.org
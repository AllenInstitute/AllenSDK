# DOCKER-VERSION 1.11.1
#
# docker build --tag alleninstitute/allensdk:anaconda .
# docker run -it alleninstitute/allensdk:anaconda /bin/bash
#
FROM continuumio/anaconda

MAINTAINER Tim Fliss <timf@alleninstitute.org>

#neuron installation
RUN cd /root; wget http://www.neuron.yale.edu/ftp/neuron/versions/v7.4/v7.4.rel-1370/nrn-7.4.rel-1370.x86_64.deb 
RUN cd /root; \
    dpkg -i nrn-7.4.rel-1370.x86_64.deb; \
    apt-get install -f; \
    dpkg -i nrn-7.4.rel-1370.x86_64.deb 

RUN pip install allensdk
RUN pip install mock

# DOCKER-VERSION 1.11.1
#
# docker build --tag alleninstitute/allensdk:anaconda3_neuron_1370 .
# docker run -it alleninstitute/allensdk:anaconda3_neuron_1370 /bin/bash
#
FROM continuumio/anaconda3

MAINTAINER David Feng <davidf@alleninstitute.org>

# neuron installation
WORKDIR root
COPY shared/conda_27.sh ./shared/
RUN /bin/bash shared/conda_27.sh

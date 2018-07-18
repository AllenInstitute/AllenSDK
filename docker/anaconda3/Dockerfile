# DOCKER-VERSION 1.11.1
#
# docker build --tag alleninstitute/allensdk:anaconda3 .
# docker run -it alleninstitute/allensdk:anaconda3 /bin/bash
#
FROM continuumio/miniconda3

MAINTAINER David Feng <davidf@alleninstitute.org>

# neuron installation
WORKDIR root
COPY shared/apt_get_dependencies.sh ./shared/
RUN /bin/bash shared/apt_get_dependencies.sh
COPY shared/conda_27.sh ./shared/
RUN /bin/bash shared/conda_27.sh
COPY shared/conda_36.sh ./shared/
RUN /bin/bash shared/conda_36.sh

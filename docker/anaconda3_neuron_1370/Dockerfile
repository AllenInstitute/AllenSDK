# DOCKER-VERSION 1.11.1
#
# docker build --tag alleninstitute/allensdk:anaconda_neuron_1370 .
# docker run -it alleninstitute/allensdk:anaconda_neuron_1370 /bin/bash
#
FROM continuumio/anaconda3

MAINTAINER Tim Fliss <timf@alleninstitute.org>

# neuron installation
WORKDIR root
COPY shared shared
COPY requirements.txt test_requirements.txt ./
RUN /bin/bash shared/apt_get_dependencies.sh
RUN /bin/bash shared/conda_27.sh
RUN /bin/bash --login shared/jupyterhub.sh
RUN /bin/bash --login shared/build_nrn.sh
RUN /bin/bash --login shared/tensorflow.sh
RUN /bin/bash --login shared/users.sh

RUN shared/build_DiPDE.sh -e py27
RUN shared/build_NEST.sh -e py27
RUN shared/build_NEURON.sh -e py27
#RUN shared/build_Tensorflow.sh -e py27

ENV PYTHONPATH /usr/local/nrn/lib/python
ENV PATH /usr/local/nrn/x86_64/bin:$PATH
ENV TEST_API_ENDPOINT http://api.brain-map.org
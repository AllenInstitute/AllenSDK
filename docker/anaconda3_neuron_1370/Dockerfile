# DOCKER-VERSION 1.11.1
#
# docker build --tag alleninstitute/allensdk:anaconda3_neuron_1370 .
# docker run -it alleninstitute/allensdk:anaconda3_neuron_1370 /bin/bash
#
FROM continuumio/anaconda3

MAINTAINER Tim Fliss <timf@alleninstitute.org>

# neuron installation
WORKDIR root
COPY requirements.txt test_requirements.txt ./
COPY shared shared/apt_get_dependencies.sh
RUN /bin/bash shared/apt_get_dependencies.sh
COPY shared shared/conda_27.sh
RUN /bin/bash shared/conda_27.sh
COPY shared shared/allensdk_install.sh
RUN /bin/bash --login shared/allensdk_install.sh
COPY shared shared/jupyterhub.sh
RUN /bin/bash --login shared/jupyterhub.sh
COPY shared shared/build_nrn.sh
RUN /bin/bash --login shared/build_nrn.sh
COPY shared shared/tensorflow.sh
RUN /bin/bash --login shared/tensorflow.sh
COPY shared shared/aws_dependencies.sh
RUN /bin/bash --login shared/aws_dependencies.sh
COPY shared shared/users.sh
RUN /bin/bash --login shared/users.sh

COPY shared/build_DiPDE.sh shared/
#RUN shared/build_DiPDE.sh -e py27
COPY shared/build_NEST.sh shared/
RUN shared/build_NEST.sh -e py27
COPY shared/build_NEURON.sh shared/
#RUN shared/build_NEURON.sh -e py27
COPY shared/build_Tensorflow.sh shared/
#RUN shared/build_Tensorflow.sh -e py27

RUN /bin/bash shared/aws_dependencies.sh

ENV PYTHONPATH /usr/local/nrn/lib/python
ENV PATH /usr/local/nrn/x86_64/bin:$PATH
ENV TEST_API_ENDPOINT http://api.brain-map.org
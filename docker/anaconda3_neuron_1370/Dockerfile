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
COPY shared/* ./shared/
RUN /bin/bash shared/apt_get_dependencies.sh; \
 /bin/bash shared/conda_27.sh; \
 /bin/bash --login shared/allensdk_install.sh; \
 /bin/bash --login shared/jupyterhub.sh; \
 /bin/bash --login shared/build_nrn.sh; \
 /bin/bash --login shared/tensorflow.sh; \
 /bin/bash --login shared/aws_dependencies.sh
 mkdir -p /root/build_nest && /bin/bash --login shared/build_NEST.sh -b /root/build_nest -e py27; \
 /bin/bash shared/aws_dependencies.sh
#RUN mkdir -p /root/build_dipde && shared/build_DiPDE.sh -b /root/build_dipde -e py27
#RUN mkdir -p /root/build_neuron && shared/build_NEURON.sh -b /root/build_neuron -e py27
#RUN mkdir -p /root/build_tensorflow && shared/build_Tensorflow.sh -b /root/build_tensorflow -e py27


ENV PYTHONPATH /usr/local/nrn/lib/python
ENV PATH /usr/local/nrn/x86_64/bin:$PATH
ENV TEST_API_ENDPOINT http://api.brain-map.org
# DOCKER-VERSION 1.11.1
#
# docker build --tag alleninstitute/allensdk:anaconda3_neuron_1370 .
# docker run -it alleninstitute/allensdk:anaconda3_neuron_1370 /bin/bash
#
FROM continuumio/anaconda3

MAINTAINER Tim Fliss <timf@alleninstitute.org>

# neuron installation
WORKDIR root
COPY shared/apt_get_dependencies.sh shared/conda_27.sh shared/tensorflow.sh shared/aws_dependencies.sh shared/build_nrn.sh shared/jupyterhub.sh ./shared/
RUN /bin/bash shared/apt_get_dependencies.sh; \
 /bin/bash shared/conda_27.sh; \
 /bin/bash --login shared/tensorflow.sh; \
 /bin/bash --login shared/aws_dependencies.sh; \
 mkdir -p /root/build_nest && /bin/bash --login shared/build_NEST.sh -b /root/build_nest -e py27; \
 /bin/bash --login shared/build_nrn.sh; \
 /bin/bash --login shared/jupyterhub.sh
COPY requirements.txt test_requirements.txt ./
COPY shared/allensdk_install.sh shared/build_DiPDE.sh shared/build_NEURON.sh shared/build_Tensorflow.sh ./shared/
RUN /bin/bash --login shared/allensdk_install.sh
#RUN mkdir -p /root/build_dipde && shared/build_DiPDE.sh -b /root/build_dipde -e py27
#RUN mkdir -p /root/build_neuron && shared/build_NEURON.sh -b /root/build_neuron -e py27
#RUN mkdir -p /root/build_tensorflow && shared/build_Tensorflow.sh -b /root/build_tensorflow -e py27
COPY shared/modelingsdk_install.sh ./shared/
COPY modelingsdk ./
RUN /bin/bash shared/modelingsdk_install.sh


ENV PYTHONPATH /usr/local/nrn/lib/python
ENV PATH /usr/local/nrn/x86_64/bin:$PATH
ENV TEST_API_ENDPOINT http://api.brain-map.org
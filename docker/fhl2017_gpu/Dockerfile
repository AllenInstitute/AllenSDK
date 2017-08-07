# DOCKER-VERSION 1.11.1
#
# docker build --tag alleninstitute/allensdk:anaconda3_neuron_1370 .
# docker run -it alleninstitute/allensdk:anaconda3_neuron_1370 /bin/bash
#
FROM gw000/debian-cuda

MAINTAINER Tim Fliss <timf@alleninstitute.org>

# neuron installation
WORKDIR root
COPY requirements.txt test_requirements.txt ./
COPY shared/apt_get_dependencies.sh shared/
RUN /bin/bash shared/apt_get_dependencies.sh
COPY shared/install_miniconda.sh shared/
RUN /bin/bash shared/install_miniconda.sh
ENV PATH /opt/conda/bin:$PATH
COPY shared/conda_27.sh shared/
RUN /bin/bash shared/conda_27.sh
COPY shared/allensdk_install.sh shared/
RUN /bin/bash shared/allensdk_install.sh
COPY shared/jupyterhub.sh shared/
RUN /bin/bash shared/jupyterhub.sh
COPY shared/build_nrn.sh shared/
#RUN /bin/bash shared/build_nrn.sh
COPY shared/tensorflow_gpu.sh shared/
RUN /bin/bash shared/tensorflow_gpu.sh
COPY shared/aws_dependencies.sh shared/
RUN /bin/bash shared/aws_dependencies.sh
#COPY shared/users.sh shared/
#RUN /bin/bash shared/users.sh

COPY shared/build_DiPDE.sh shared/
#RUN mkdir -p /root/build_dipde && shared/build_DiPDE.sh -b /root/build_dipde -e py27
COPY shared/build_NEST.sh shared/
#RUN mkdir -p /root/build_nest && /bin/bash --login shared/build_NEST.sh -b /root/build_nest -e py27
COPY shared/build_NEURON.sh shared/
#RUN mkdir -p /root/build_neuron && shared/build_NEURON.sh -b /root/build_neuron -e py27
COPY shared/build_Tensorflow.sh shared/
#RUN mkdir -p /root/build_tensorflow && shared/build_Tensorflow.sh -b /root/build_tensorflow -e py27

RUN /bin/bash shared/aws_dependencies.sh

ENV PYTHONPATH /usr/local/nrn/lib/python
ENV PATH /usr/local/nrn/x86_64/bin:$PATH
ENV TEST_API_ENDPOINT http://api.brain-map.org

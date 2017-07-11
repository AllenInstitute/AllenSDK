# DOCKER-VERSION 1.11.1
#
# docker build --tag alleninstitute/allensdk .
# docker run -it alleninstitute/allensdk /bin/bash
#
FROM alleninstitute/anaconda_neuron_1370

MAINTAINER Tim Fliss <timf@alleninstitute.org>

COPY . ./allensdk/
RUN  pip install ./allensdk
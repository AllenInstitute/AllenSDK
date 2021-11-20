# DOCKER-VERSION 1.11.1
#
# docker build --tag alleninstitute/fhl2017:gpu .
# docker run -it alleninstitute/fhl2017:gpu /bin/bash
#
FROM alleninstitute/anaconda3_neuron_1370

MAINTAINER Tim Fliss <timf@alleninstitute.org>

# tensorflow installation
WORKDIR root
COPY shared ./shared/
RUN /bin/bash shared/install_torch.sh

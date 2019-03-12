#
# A Docker image for running allensdk with neuronal network simulations
#
# docker build --tag alleninstitute/centos7_base
#
FROM centos:6
MAINTAINER Tim Fliss <timf@alleninstitute.org>

COPY yum_install.sh ./
RUN /bin/bash yum_install.sh

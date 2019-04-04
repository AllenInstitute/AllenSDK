# DOCKER-VERSION 0.3.4
#
# docker build --tag alleninstitute/lims2_modules
# docker run -it alleninstitute/lims2_modules /bin/bash
#
FROM alleninstitute/lims2_base

MAINTAINER Tim Fliss <timf@alleninstitute.org>

USER root
RUN apt-get install -yq git
RUN mkdir -p /shared/bioapps/infoapps; \
    cd /shared/bioapps/infoapps; \
    git clone --depth 1 http://timf@stash.corp.alleninstitute.org/scm/inf/lims2_modules.git

# Install lims2_modules and allensdk+internal
RUN cd /shared/bioapps/infoapps/lims2_modules/lib; \
    wget -O allensdk.zip http://stash.corp.alleninstitute.org/plugins/servlet/archive/projects/INF/repos/allensdk?at=refs%2Fheads%2Flims2_modules; \
    unzip allensdk.zip; \
    rm allensdk.zip; \
    mkdir -p allensdk/internal; \
    cd allensdk/internal; \
    wget -O allensdk_internal.zip http://stash.corp.alleninstitute.org/plugins/servlet/archive/projects/INF/repos/allensdk_internal?at=refs%2Fheads%2Fdev; \
    unzip allensdk_internal.zip; \
    rm allensdk_internal.zip; \
    ln -s /usr/local/miniconda/envs/lims python2.7

WORKDIR /root
COPY lint.sh .
RUN apt-get install -yq vim ncurses-dev

# Install ainwb
RUN cd /shared/bioapps/infoapps/lims2_modules/lib; \
    wget -O ainwb.zip http://stash.corp.alleninstitute.org/plugins/servlet/archive/projects/INF/repos/ainwb?at=refs%2Fheads%2Fdev; \
    unzip ainwb.zip; \
    rm ainwb.zip

# TODO Install Deap, etc.
RUN pip install deap

ENV PYTHONPATH=/shared/bioapps/infoapps/lims2_modules/lib:/shared/bioapps/infoapps/lims2_modules/lib/ainwb:/shared/utils.x86_64/nrn-7.4-1370/lib/python:$PYTHONPATH

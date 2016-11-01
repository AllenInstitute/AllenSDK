# DOCKER-VERSION 1.11.1
#
# docker build --tag alleninstitute/allensdk:anaconda .
# docker run -it alleninstitute/allensdk:anaconda /bin/bash
#
FROM alleninstitute/allensdk:anaconda_neuron_1370

MAINTAINER Tim Fliss <timf@alleninstitute.org>

RUN mkdir allensdk; \
    cd allensdk; \
    wget -O allensdk.zip http://stash.corp.alleninstitute.org/plugins/servlet/archive/projects/INF/repos/allensdk?at=refs%2Fheads%2Fdev; \
    unzip allensdk.zip; \
    rm allensdk.zip; \
    pip install .; \
    pip install -r test_requirements.txt
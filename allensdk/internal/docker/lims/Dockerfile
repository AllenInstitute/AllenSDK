# DOCKER-VERSION 0.3.4
#
# docker build --tag alleninstitute/lims
# docker run -it alleninstitute/lims /bin/bash
#
FROM alleninstitute/lims2_modules

MAINTAINER Tim Fliss <timf@alleninstitute.org>

USER docker
WORKDIR /home/docker
RUN touch .profile
ENV HOME /home/docker
ENV GEM_HOME /home/docker/.gems

RUN mkdir lims; \
    cd lims; \
    wget -O lims.zip http://stash.corp.alleninstitute.org/plugins/servlet/archive/projects/TECH/repos/lims?at=refs%2Fheads%2F16.40_L; \
    unzip lims.zip; \
    echo rm lims.zip 

WORKDIR /home/docker/lims
USER docker
ENV PATH /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/home/docker/.gems/bin:/usr/local/miniconda/bin
RUN bundle install

USER root
RUN mkdir -p /shared/bioapps/infoapps/lims2_modules/lib/allensdk; \
    cd /shared/bioapps/infoapps/lims2_modules/lib/allensdk; \
    wget -O allensdk.zip http://stash.corp.alleninstitute.org/plugins/servlet/archive/projects/INF/repos/allensdk?at=refs%2Fheads%2Fdev; \
    unzip allensdk.zip; \
    rm allensdk.zip; \
    mkdir -p allensdk/internal; \
    cd allensdk/internal; \
    wget -O allensdk_internal.zip http://stash.corp.alleninstitute.org/plugins/servlet/archive/projects/INF/repos/allensdk_internal?at=refs%2Fheads%2Fdev; \
    unzip allensdk_internal.zip; \
    rm allensdk_internal.zip
ENV PYTHONPATH /shared/bioapps/infoapps/lims2_modules/lib/allensdk:$PYTHONPATH

COPY database.yml /home/docker/lims/config
COPY neuronal_model_run_files.py /home/docker
ENV  PATH /home/docker/.gems/bin:$PATH
USER docker
CMD  ["/bin/bash", "-c", "cd /home/docker/lims && rails s"]

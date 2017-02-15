# DOCKER-VERSION 1.11.1
#
# docker build --tag alleninstitute/allensdk:anaconda .
# docker run -it alleninstitute/allensdk:anaconda /bin/bash
#
FROM alleninstitute/allensdk:anaconda_neuron_1370

MAINTAINER Tim Fliss <timf@alleninstitute.org>

COPY env.sh git_clone.sh ./

RUN /bin/bash -c "\
  chmod +x git_clone.sh && \
  source env.sh && \
  ./git_clone.sh && \
  cd allensdk && \
  pip install . && \
  pip install -r test_requirements.txt"

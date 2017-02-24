#
# A Docker image for running allensdk with neuronal network simulations
#
# docker build --tag alleninstitute/allensdk:neuralensemble .
# docker run -it alleninstitute/allensdk:neuralensemble /bin/bash
# source ~/env/neurosci/bin/activate
#
FROM alleninstitute/allensdk:neuralensemble_py3_1370
MAINTAINER Tim Fliss <timf@alleninstitute.org>

COPY env.sh git_clone.sh ./

RUN /bin/bash -c "\
  chmod +x git_clone.sh && \
  source env.sh && \
  ./git_clone.sh && \
  cd allensdk && \
  pip install . && \
  pip install -r test_requirements.txt"

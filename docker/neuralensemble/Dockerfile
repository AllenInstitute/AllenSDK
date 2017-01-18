#
# A Docker image for running allensdk with neuronal network simulations
#
# docker build --tag alleninstitute/allensdk:neuralensemble .
# docker run -it alleninstitute/allensdk:neuralensemble /bin/bash
# source ~/env/neurosci/bin/activate
#
FROM alleninstitute/allensdk:neuralensemble_py3_1370
MAINTAINER Tim Fliss <timf@alleninstitute.org>

# TODO: replace w/ install from Github
RUN mkdir allensdk; \
    cd allensdk; \
    wget -O allensdk.zip http://stash.corp.alleninstitute.org/plugins/servlet/archive/projects/INF/repos/allensdk?at=refs%2Fheads%2Fdev; \
    unzip allensdk.zip; \
    rm allensdk.zip; \
    $VENV/bin/pip install .; \
    $VENV/bin/pip install -r test_requirements.txt; \
    $VENV/bin/pip install jupyter; \
    $VENV/bin/jupyter-notebook --generate-config
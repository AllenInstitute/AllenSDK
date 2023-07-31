#
# To deploy an image for use on bamboo, use the following command:
# docker buildx build --platform linux/amd64,linux/arm64 -t alleninstitutepika/allensdk_anaconda3 --push docker/anaconda3
#
# To pull the container:
# docker run -it alleninstitutepika/anaconda3 /bin/bash
# NOTE: This container is designed for testing only and does not have the
# AllenSDK installed.
FROM continuumio/miniconda3:23.5.2-0

LABEL maintainer="nicholas.mei@alleninstitute.org"

RUN apt-get update \
    && apt-get install -y \
        automake \
        libopenjp2-7 \
        make \
        pkg-config \
        rsync \
    && rm -rf /var/lib/apt/lists/*

RUN conda update -y conda

RUN conda create -y --name py37 python=3.7 ipykernel \
    && conda clean --index-cache --tarballs

RUN conda create -y --name py38 python=3.8 ipykernel numpy \
    && conda clean --index-cache --tarballs

RUN conda create -y --name py39 python=3.9 ipykernel \
    && conda clean --index-cache --tarballs

RUN conda create -y --name py310 python=3.10 ipykernel \
    && conda clean --index-cache --tarballs

RUN conda create -y --name py311 python=3.11 ipykernel \
    && conda clean --index-cache --tarballs

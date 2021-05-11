# docker build --tag alleninstitute/allensdk:anaconda3 .
# docker run -it alleninstitute/allensdk:anaconda3 /bin/bash
#
FROM continuumio/miniconda3:4.8.2

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

RUN conda create -y --name py36 python=3.6 ipykernel \
    && conda clean --index-cache --tarballs

RUN conda create -y --name py37 python=3.7 ipykernel \
    && conda clean --index-cache --tarballs

RUN conda create -y --name py38 python=3.8 ipykernel numpy \
    && conda clean --index-cache --tarballs

RUN conda create -y --name py39 python=3.9 ipykernel \
    && conda clean --index-cache --tarballs

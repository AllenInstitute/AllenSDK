# docker build --tag alleninstitute/allensdk:anaconda3_test .
# docker run -it alleninstitute/allensdk:anaconda3_test /bin/bash
#
FROM continuumio/miniconda3:4.8.2

RUN apt-get update \
    && apt-get install -y \
        automake \
        libopenjp2-7 \
        make \
        pkg-config \
        rsync \
    && rm -rf /var/lib/apt/lists/*

RUN conda update -y conda

RUN conda create -y --name py38 python=3.8 ipykernel numpy\
    && conda clean --index-cache --tarballs

ADD . /root/AllenSDK
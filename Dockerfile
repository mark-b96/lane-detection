ARG CUDA_VER=11.3.0
ARG UBUNTU_VER=20.04
ARG CUDNN_VER=8

FROM nvidia/cuda:${CUDA_VER}-cudnn${CUDNN_VER}-runtime-ubuntu${UBUNTU_VER}

ENV PYTHON_VER=3.8
ENV DEBIAN_FRONTEND=noninteractive

COPY requirements.txt .

RUN apt-get update && apt-get upgrade -y \
    apt-utils \
    libgl1 \
    libglib2.0-0 \
	python3-pip \
	python${PYTHON_VER} \
	python${PYTHON_VER}-dev

RUN	pip install -r requirements.txt


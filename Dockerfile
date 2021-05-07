FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && \
    apt-get install -y git vim libgtk2.0-dev && \
    rm -rf /var/cache/apk/*

RUN pip --no-cache-dir install Cython

COPY requirements.txt /workspace

RUN pip --no-cache-dir install -r /workspace/requirements.txt

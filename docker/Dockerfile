# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu20.04
ENV TZ=TR GMS_HOME=/home/root/cgm-gm
SHELL ["/bin/bash", "-c"]

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt install -y python3.10 python3.10-venv python3.10-dev git curl
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN rm /usr/bin/python3
RUN ln -s /usr/bin/python3.10 /usr/bin/python

RUN git clone https://github.com/CengizPoyraz/cgm-gm.git $GMS_HOME
RUN pip install -r $GMS_HOME/requirements.txt

# in case you prefer to use venv
# RUN python -m venv .venv 
# RUN source .venv/bin/activate


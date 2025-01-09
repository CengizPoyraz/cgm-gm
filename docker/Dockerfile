# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu20.04
ENV TZ=TR \
    DEBIAN_FRONTEND=noninteractive \
    GMS_HOME=/home/root/cgm-gm \
    GMS_GIT_URL=https://github.com/CengizPoyraz/cgm-gm.git
WORKDIR $GMS_HOME
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt upgrade -y
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa 
RUN apt-get update && apt install -y python3.10 python3.10-venv python3.10-dev git curl nano
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN rm /usr/bin/python3
RUN ln -s /usr/bin/python3.10 /usr/bin/python
RUN (type -p wget >/dev/null || ( apt update &&  apt-get install wget -y)) \
	&&  mkdir -p -m 755 /etc/apt/keyrings \
        && out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        && cat $out |  tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
	&&  chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
	&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" |  tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
	&&  apt update \
	&&  apt install gh -y

ENV DEBIAN_FRONTEND= 

RUN git clone $GMS_GIT_URL $GMS_HOME
RUN pip install -r $GMS_HOME/requirements.txt
# in case you prefer to use venv
# RUN python -m venv $GMS_HOME/.venv 
# RUN source $GMS_HOME/.venv/bin/activate


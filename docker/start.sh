#!/bin/bash
docker run -it --mount type=bind,src=/home/gms/docker/shared,target=/home/root/host --gpus all cpoyraz/gms:v3.4 bash
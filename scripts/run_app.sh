#!/bin/bash

AWS_ACCOUNT_ID=739366754163
AWS_REGION=eu-west-2
ECR_DOCKER_IMG=lane-det-img:latest

docker pull ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_DOCKER_IMG}

if ! type "nvidia-smi" > /dev/null; then
    echo "Running on CPU"
    docker run -it -v /home/${SUDO_USER}:/home/${SUDO_USER} --hostname $HOSTNAME --name lane-det-container markb96/lane-det-repo:myfirstpush
else
    echo "Running on GPU"
    docker run --gpus all -it -v /home/${SUDO_USER}:/home/${SUDO_USER} --hostname $HOSTNAME --name lane-det-container markb96/lane-det-repo:myfirstpush
fi

#!/bin/bash

docker pull markb96/lane-det-repo:myfirstpush

if ! type "nvidia-smi" > /dev/null; then
    echo "Running on CPU"
    docker run -it -v /home/${SUDO_USER}:/home/${SUDO_USER} --hostname $HOSTNAME --name lane-det-container markb96/lane-det-repo:myfirstpush
else
    echo "Running on GPU"
    docker run --gpus all -it -v /home/${SUDO_USER}:/home/${SUDO_USER} --hostname $HOSTNAME --name lane-det-container markb96/lane-det-repo:myfirstpush
fi

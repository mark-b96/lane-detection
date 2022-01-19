#!bin/bash

if ! type "nvidia-smi" > /dev/null; then
    echo "Running on CPU"
    docker run -it -v /home/ubuntu:/home/ubuntu --hostname $HOSTNAME --name lane-det-container lane-det-img
else
    echo "Running on GPU"
    docker run --gpus all -it -v /home/ubuntu:/home/ubuntu --hostname $HOSTNAME --name lane-det-container lane-det-img
fi


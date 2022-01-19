#!bin/bash

docker run -it -v /home/ubuntu:/home/ubuntu --hostname $HOSTNAME --name lane-det-container lane-det-img

#!/bin/bash

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=eu-west-2
ECR_DOCKER_IMG=lane-det-img:latest

#curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
#unzip awscliv2.zip
#./aws/install

#docker login -u AWS -p $(aws ecr get-login-password --region ${AWS_REGION}) \
# ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

#docker pull ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_DOCKER_IMG}

SCRIPT_RELATIVE_DIR=$(dirname "${BASH_SOURCE[0]}")
cd $SCRIPT_RELATIVE_DIR
cd ../

if ! type "nvidia-smi" > /dev/null; then
    echo "Running on CPU"
    docker run --rm -v /home/${SUDO_USER}:/home/${SUDO_USER} -v /opt:/opt --hostname $HOSTNAME \
          --name -e USER=$USER lane-det-container ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_DOCKER_IMG} \
           /bin/bash -c "cd ${PWD}; python3.8 lane_detector.py --train True --weights_dir /home/${USER}/weights --training_images_dir /home/${USER}/training_data/images --training_labels_dir /home/${USER}/training_data/labels"
else
    echo "Running on GPU"
    docker run --rm --gpus all -v /home/${SUDO_USER}:/home/${SUDO_USER} -v /opt:/opt --hostname $HOSTNAME \
          --name -e USER=$USER lane-det-container ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_DOCKER_IMG} \
           /bin/bash -c "cd ${PWD}; python3.8 lane_detector.py --train True --weights_dir /home/${USER}/weights --training_images_dir /home/${USER}/training_data/images --training_labels_dir /home/${USER}/training_data/labels"
fi

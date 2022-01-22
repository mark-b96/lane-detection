#!/bin/bash

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=eu-west-2
ECR_DOCKER_IMG=lane-det-img:latest

#curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
#unzip awscliv2.zip
#./aws/install

#docker login -u AWS -p $(aws ecr get-login-password --region ${AWS_REGION}) ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

#docker pull ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_DOCKER_IMG}

if ! type "nvidia-smi" > /dev/null; then
    echo "Running on CPU"
    docker run -rm -v /home/${SUDO_USER}:/home/${SUDO_USER} --hostname $HOSTNAME --name lane-det-container ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_DOCKER_IMG} /bin/bash -c "cd ${PWD}; python3.8 lane_detector.py"
else
    echo "Running on GPU"
    docker run -rm --gpus all -v /home/${SUDO_USER}:/home/${SUDO_USER} --hostname $HOSTNAME --name lane-det-container ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_DOCKER_IMG} /bin/bash -c "cd ${PWD}; python3.8 lane_detector.py"
fi

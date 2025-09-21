#!/bin/bash
set -x #echo on

# Install nvidia
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm -r cuda-keyring_1.1-1_all.deb

# Update system
sudo add-apt-repository ppa:quentiumyt/nvtop -y
sudo apt update -y
sudo apt upgrade -y
sudo apt-get install build-essential cuda-toolkit nodejs npm nvidia-gds btop nvtop -y

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install aws cli
# curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
# unzip -u awscliv2.zip
# rm -r awscliv2.zip
# sudo ./aws/install

# Setup AWS
# aws configure
# aws s3 cp s3://thesis2025-lenguyen/wmdp-corpora.zip ./wmdp-corpora.zip
# unzip wmdp-corpora.zip
# rm -r __MACOSX && rm wmdp-corpora.zip && mkdir rmu/data && mv wmdp-corpora/* rmu/data && rm -r wmdp-corpora

# Reboot
sudo reboot

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

# Reboot
sudo reboot

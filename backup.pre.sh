#!/bin/bash
# May need to uncomment and update to find current packages
# apt-get update

# Required for demo script! #
#conda create -y -p /home/workspace/tools/py2p7 python=2.7 anaconda
source activate /home/workspace/tools/py2p7
pip install scikit-video

# Add your desired packages for each workspace initialization
#          Add here!          #
pip install pillow keras sacred numpy matplotlib

cd tools
sudo dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-ga-cuda9.0-trt3.0-20171128/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda-libraries-9-0

sudo dpkg -i libcudnn7_7.1.4.18-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-dev_7.1.4.18-1+cuda9.0_amd64.deb
cd ..

pip install --upgrade tensorflow-gpu
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/
python Keras-FCN/utils/transfer_FCN.py ResNet50


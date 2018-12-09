#!/bin/bash

sudo apt install python3-pip jupyter gcc \
     libjpeg62-dev libavcodec-dev libswscale-dev \
     libffi-dev gstreamer1.0-opencv

sudo pip3 install opencv-python numpy matplotlib \
     Cython imgaug 'setuptools<=39.1.0' wheel tensorflow vi3o

git clone --depth=1 https://github.com/ahmedbilal/pyflow.git

git clone --depth=1 https://github.com/ahmedbilal/coco.git

git clone --depth=1 https://github.com/matterport/Mask_RCNN.git

echo "Using ${PYTHON}"

sudo PYTHON=${PYTHON} make -C coco/PythonAPI
sudo rm -rf coco

echo "Setup Complete"

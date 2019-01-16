#!/bin/bash

sudo apt install python3-pip jupyter gcc \
     libjpeg62-dev libavcodec-dev libswscale-dev \
     libffi-dev gstreamer1.0-opencv ffmpeg

git clone --depth=1 https://github.com/ahmedbilal/coco.git

git clone --depth=1 https://github.com/matterport/Mask_RCNN.git

git clone --depth=1 https://github.com/ahmedbilal/yolov3.git
echo "Using ${PYTHON}"

sudo PYTHON=${PYTHON} make -C coco/PythonAPI
sudo rm -rf coco

echo "Setup Complete"

# Video Summarization
Video summarization is used to create a summarized video with user specified features from the original video. In this project, we will develop a dynamic video summarization (simultaneous presentation of events that will enable the review of hours of video footage in just minutes). The main goal of this project is to save time when extracting information from long videos. We are doing this by shifting events from their original time interval to another time interval when no other activity is happening at that spatial location. We are selecting some long videos and also obtaining some CCTV camera footages from live CCTV streamed from YouTube. The output of our program would be a summarized video created from a lengthy original video with user specified features. 


## Development Environment Setup
Run the following command to make your development environment identical to that of Ahmad Bilal Khalid (ABK).
### Ubuntu 18.04 or 18.10

Run the following commands one by one with patience as they can take
some time.
```bash
sudo apt install python3-pip jupyter gcc \
    libjpeg62-dev libavcodec-dev libswscale-dev \
    libffi-dev gstreamer1.0-opencv

sudo pip3 install opencv-python numpy matplotlib \
Cython imgaug 'setuptools<=39.1.0' wheel tensorflow vi3o

cd ~/Desktop

git clone --depth=1 https://github.com/waleedka/coco

nano coco/PythonAPI/Makefile
```
Replace, **python** with **python3** and Press **Ctrl + x** and then **y** to save file
```bash
sudo apt install 

sudo make install -C coco/PythonAPI

git clone --depth=1 https://github.com/ahmedbilal/Video-Summarization.git

git clone --depth=1 https://github.com/matterport/Mask_RCNN.git
```


## References
1. https://en.wikipedia.org/wiki/Video_synopsis
2. http://www.vision.huji.ac.il/video-synopsis/iccv07-webcam.pdf
3. http://www.vision.huji.ac.il/video-synopsis/pami08-synopsis.pdf
4. http://www.vision.huji.ac.il/video-synopsis/cvpr06-synopsis.pdf

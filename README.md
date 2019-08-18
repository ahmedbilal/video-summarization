# Video Summarization
Video summarization is used to create a summarized video with user specified features from the original video. In this project, we will develop a dynamic video summarization (simultaneous presentation of events that will enable the review of hours of video footage in just minutes). The main goal of this project is to save time when extracting information from long videos. We do this by shifting events from their original time interval to another time interval when no other activity is happening at that spatial location. We are selecting some long videos and also obtaining some CCTV camera footages from live CCTV streamed from YouTube. The output of our program would be a summarized video created from a lengthy original video with user specified features. 

## License
This project is licensed under GNU GPL v3. You can read about in *LICENSE* file found in the root of this repository. TLDR: You can do almost anything with it except to distribute closed source versions.


## Development Environment Setup
Run the following command to make your development environment identical to that of Ahmad Bilal Khalid (ABK).

### Manjaro

1. Enable AUR packages
2. Install **python36** from AUR
3. Install **pip** for python3.6

Run the following command if you have Nvidia Cuda Enabled GPU
```bash
sudo ln -s /opt/cuda/ /usr/local/cuda
```

Run these command one by one
```bash
git clone https://github.com/ahmedbilal/video_summarization.git --depth=1
cd video_summarization
git clone https://github.com/ahmedbilal/yolov3.git
git clone https://github.com/matterport/Mask_RCNN.git
cd yolov3 && make
mkdir weight
cd weight
wget https://pjreddie.com/media/files/yolov3.weights
cd ../../

sudo pip3.6 install -r requirements.txt

python3 main.py
```


## Todo
- \[x] Create Summarized Video
- \[ ] Fix Issues with Summarized Video
- \[ ] Helmet Detection
- \[ ] Speed Detection
- \[ ] Density Detection

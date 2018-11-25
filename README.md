# Video Summarization
![Video Summarization](Resources/cover.jpg)
**Cover Image from [Digital Security Magazine](https://www.digitalsecuritymagazine.com)**

Video summarization is used to create a summarized video with user specified features from the original video. In this project, we will develop a dynamic video summarization (simultaneous presentation of events that will enable the review of hours of video footage in just minutes). The main goal of this project is to save time when extracting information from long videos. We are doing this by shifting events from their original time interval to another time interval when no other activity is happening at that spatial location. We are selecting some long videos and also obtaining some CCTV camera footages from live CCTV streamed from YouTube. The output of our program would be a summarized video created from a lengthy original video with user specified features. 


## Development Environment Setup
Run the following command to make your development environment identical to that of Ahmad Bilal Khalid (ABK).
### Ubuntu 18.04

Run the following commands one by one.
```bash
sudo apt install python3-pip

sudo apt install jupyter

sudo pip3 install opencv-python numpy matplotlib Cython imgaug 'setuptools<=39.1.0' wheel tensorflow

cd ~/Desktop

git clone https://github.com/waleedka/coco

nano coco/PythonAPI/Makefile
```
Replace, **python** with **python3** and Press **Ctrl + x** and then **y** to save file
```bash
sudo apt install gcc

sudo make install -C coco/PythonAPI

git clone https://github.com/ahmedbilal/Video-Summarization.git

git clone https://github.com/matterport/Mask_RCNN.git
```

## Usage
### Jupyter Notebook
**Note** Notebooks are often outdated. They are just used to develop initial working program. For latest and greatest implementation refer to python files (with .py extension)


To run the jupyter notebooks (files with .ipynb extension) run the following command.

```bash
jupyter notebook
```

1. Now, browse and open your .ipynb file.
2. It is opened but not trusted to run on your computer.
3. To trust click on **Not Trusted** button on the top right corner of your screen.
4. A dialog will appear now click the red **Trust** button.
5. You have successfully trusted the notebook.
6. To run it click on **Kernel->Restart and Run all->Restart and Run all Cell** from top menu.

## References
1. https://en.wikipedia.org/wiki/Video_synopsis
2. http://www.vision.huji.ac.il/video-synopsis/iccv07-webcam.pdf
3. http://www.vision.huji.ac.il/video-synopsis/pami08-synopsis.pdf
4. http://www.vision.huji.ac.il/video-synopsis/cvpr06-synopsis.pdf

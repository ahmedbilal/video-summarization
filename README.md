# Video Summarization
Video summarization is used to create a summarized video with user specified features from the original video. In this project, we will develop a dynamic video summarization (simultaneous presentation of events that will enable the review of hours of video footage in just minutes). The main goal of this project is to save time when extracting information from long videos. We are doing this by shifting events from their original time interval to another time interval when no other activity is happening at that spatial location. We are selecting some long videos and also obtaining some CCTV camera footages from live CCTV streamed from YouTube. The output of our program would be a summarized video created from a lengthy original video with user specified features. 

## Software
1. VLC

## Dependencies
1. NumPy
2. OpenCV

## Installation
### Ubuntu 18.04

Run the following command to install jupyter
```bash
sudo apt install jupyter
```

Run the following command to install numpy and opencv, matplotlib
```bash
sudo pip3 install opencv-python numpy matplotlib
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

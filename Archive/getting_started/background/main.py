import cv2
import numpy as np


video = cv2.VideoCapture('vid.mp4')
#captures v ideo
success, frame = video.read()
#get a frame
count = 0
alpha = 0.001
#some variable
avg1 = np.float32(frame)
avg2 = np.float32(frame)
#for manipulation
length = video.get(cv2.CAP_PROP_FRAME_COUNT)-2
#getting no of frames
while (success):
    success, f = video.read()
    temp = np.float32(f)
    cv2.accumulateWeighted(temp, avg1, alpha)
    #adding effect of a frame to avg(result) frame
    if count == length:
        cv2.imwrite("avg.jpg", avg1)
        break
    count += 1



import cv2
import numpy as np
from matplotlib import pyplot as plt


def diff_perc(image1, image2):
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(gray_image1,gray_image2)
    return np.sum(diff)/np.prod(diff.shape)


def substantial_motion_video(filename, out_filename):
    THRESHOLD = 0.5
    ADJUSTMENT = 0.7
    frames = []
    substantially_diff_frames = []

    video = cv2.VideoCapture(filename)
    frames_per_sec_in_org_video = video.get(cv2.CAP_PROP_FPS)
    success, image = video.read()
    while success:
        frames.append(image)
        success, image = video.read()

    substantially_diff_frames.append(frames[0])

    for frame in frames:
        diff = diff_perc(frame, substantially_diff_frames[-1])
        if diff > THRESHOLD:
            substantially_diff_frames.append(frame)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video = cv2.VideoWriter(out_filename, fourcc,
                            frames_per_sec_in_org_video * ADJUSTMENT, tuple(reversed(substantially_diff_frames[0].shape[:2])))

    for frame in substantially_diff_frames:
        video.write(frame)
    video.release()

    del frames
    del substantially_diff_frames

substantial_motion_video("Normal_Videos319_x264.mp4", "short.mp4")


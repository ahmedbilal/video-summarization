import os

import cv2
import numpy as np


def diff_perc(image1, image2):
    """
        Author: Ahmed Bilal Khalid

        Returns absolute difference of image1 and image2 in percentage
    """
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(gray_image1, gray_image2)
    return np.sum(diff)/np.prod(diff.shape)


class ABKVideoCapture(object):
    """
        Author: Ahmed Bilal Khalid
    """

    def __init__(self, file):
        if os.path.isfile(file):
            self.video = cv2.VideoCapture(file)
            self.frames = self.frames_generator()
            self.frames_per_second = self.video.get(cv2.CAP_PROP_FPS)
            self.total_frames = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
            width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.dimensions = width, height
        else:
            raise FileNotFoundError()

    def frames_generator(self):
        """
        Author: Ahmed Bilal Khalid

        Returns next frame from self.video"""

        # return read_status and next_frame_image
        success, image = self.video.read()
        while success:
            success, image = self.video.read()
            if success:
                yield image

    def next_frame(self):
        """
        Author: Ahmed Bilal Khalid

        Returns next frame from self.frames"""

        return next(self.frames)

    def create_substantial_motion_video(self, out_filename, fourcc_str="avc1"):
        """
        Author: Ahmed Bilal Khalid

        Creates a new svideo which only consists of those frames from original
        video that have substantial motion as compared to previously selected
        frame."""

        THRESHOLD = 0.5

        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        output_video = cv2.VideoWriter(out_filename, fourcc,
                                       self.frames_per_second,
                                       self.dimensions)

        last_frame = self.next_frame()

        for frame in self.frames:
            diff = diff_perc(last_frame, frame)
            if diff > THRESHOLD:
                last_frame = frame
                output_video.write(frame)

        output_video.release()

    def get_background(self):
        background = np.float32(cv2.cvtColor(self.next_frame(), cv2.COLOR_BGR2RGB))

        for frame in self.frames:
            background = np.add(background, cv2.cvtColor(np.float32(frame), cv2.COLOR_BGR2RGB))
        background = background / self.total_frames

        return background.astype(np.uint8)
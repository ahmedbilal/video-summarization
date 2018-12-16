import os
import pickle
import tarfile
import zlib

import cv2
import numpy as np


def diff_perc(image1, image2):
    """
        Author: Ahmed Bilal Khalid
        Contributor: None

        Returns absolute difference of image1 and image2 in percentage
    """
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(gray_image1, gray_image2)
    return np.sum(diff)/np.prod(diff.shape)


class BATRVideoCapture(object):
    __author__ = "Ahmad Bilal Khalid"
    __credits__ = None

    def __init__(self, file, offset=0):
        if os.path.isfile(file):
            self.video = cv2.VideoCapture(file)
            self.frames = self.frames_generator()
            self.frames_per_second = self.video.get(cv2.CAP_PROP_FPS)
            self.total_frames = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
            self.dimensions = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)), \
                int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video.set(cv2.CAP_PROP_POS_FRAMES, offset-1)
        else:
            raise FileNotFoundError()

    def set_offset(self, offset):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, offset-1)
    
    def frames_generator(self):
        """
        Author: Ahmed Bilal Khalid
        Contributor: None

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

        Returns next frame from self.frames
        """

        return next(self.frames)

    def create_substantial_motion_video(self, out_filename, fourcc_str="avc1"):
        """
        Author: Ahmed Bilal Khalid
        Contributor: None

        Creates a new video which only consists of those frames from original
        video that have substantial motion as compared to previously selected
        frame.
        """

        _threshold = 0.5

        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        output_video = cv2.VideoWriter(out_filename, fourcc,
                                       self.frames_per_second,
                                       self.dimensions)

        last_frame = self.next_frame()

        for frame in self.frames:
            diff = diff_perc(last_frame, frame)
            if diff > _threshold:
                last_frame = frame
                output_video.write(frame)

        output_video.release()

    def get_background(self):
        background = np.float32(cv2.cvtColor(self.next_frame(), cv2.COLOR_BGR2RGB))

        for frame in self.frames:
            background = np.add(background, cv2.cvtColor(np.float32(frame), cv2.COLOR_BGR2RGB))
        background = background / self.total_frames

        return background.astype(np.uint8)


class BATRPickle(object):
    def __init__(self, in_file=None, out_file=None):
        self.input_file = None
        self.output_file = None

        if in_file:
            filename, file_extension = os.path.splitext(in_file)
            self.input_file = tarfile.open(in_file,
                                           "r:{}".format(file_extension[1:]))  # ignoring the dot in extension
        
        if out_file:
            filename, file_extension = os.path.splitext(out_file)
            self.output_file = tarfile.open(out_file,
                                            "w:{}".format(file_extension[1:]))  # ignoring the dot in extension

    def pickle(self, obj, filename):
        """
        Author: Ahmed Bilal Khalid
        Contributor: None

        Compress obj using zlib and add that compressed obj to self.output_file
        """

        serialized_obj = pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)
        compressed = zlib.compress(serialized_obj, 9)
        with open(f"{filename}.pickled.zlibed", "wb") as f:
            f.write(compressed)        
        self.output_file.add(f"{filename}.pickled.zlibed")
        os.remove(filename)

    # def unpickle(self):
    #     """
    #     Author: Ahmed Bilal Khalid
    #     Contributor: None
    #
    #     Return an iterator to name of uncompressed and unpickled file member of self.input_file and its content
    #     """
    #
    #     members = self.input_file.getmembers()
    #     for member in members:
    #         f = self.input_file.extractfile(member)
    #         content = f.read()
    #         uncompressed = zlib.decompress(content)
    #         yield member.name, pickle.loads(uncompressed)

    def unpickle(self, filename):
        member = self.input_file.getmember("{}.pickled.zlibed".format(filename))
        f = self.input_file.extractfile(member)
        content = f.read()
        uncompressed = zlib.decompress(content)
        return pickle.loads(uncompressed)

    def __del__(self):
        """
        Author: Ahmed Bilal Khalid
        Contributor: None

        Close open compressed files
        """

        if self.input_file:
            self.input_file.close()
        
        if self.output_file:
            self.output_file.close()

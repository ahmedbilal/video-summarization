from os.path import join as joinpath
from os.path import abspath
from os import makedirs

from utilities.utils import sha1
import wget


class Video(object):
    def __init__(self, video_path, url_path=None):
        self.url_path = url_path
        self.video_path = abspath(video_path)
        self.processing_path = joinpath("processing", sha1(self.video_path))
        self.frames_path = joinpath(self.processing_path, "frames")
        self.result_path = joinpath(self.processing_path, "results")
        self.background_path = joinpath(self.result_path, "background.jpeg")
        self.detection_path = joinpath(self.result_path, "detections.tar.gz")
        self.improved_detection_path = joinpath(self.result_path, "improved_detections.tar.gz")
        self.store_path = joinpath(self.result_path, "object_store.pickled")
        self.store_data_path = joinpath(self.result_path, "store")
        self.foreground_masks_path = joinpath(self.processing_path, "foreground_masks")
        self.summarized_frames_path = joinpath(self.processing_path, "summarized_frames")


class VideoWithDir(object):
    def __init__(self, video: Video):
        self.video = video
        self.make_dir()

    def __getattr__(self, name):
        return getattr(self.video, name)

    def make_dir(self):
        """
        Make Directory needed to store components of Summarized Video
        :return:
        """
        makedirs(self.frames_path, exist_ok=True)
        makedirs(self.result_path, exist_ok=True)
        makedirs(self.store_data_path, exist_ok=True)
        makedirs(self.foreground_masks_path, exist_ok=True)
        makedirs(self.summarized_frames_path, exist_ok=True)


class VideoWithDownloader(object):
    def __init__(self, video: Video):
        self.video = video
        self.download()

    def __getattr__(self, name):
        return getattr(self.video, name)

    def download(self):
        if self.video.url_path:
            wget.download(self.video.url_path)

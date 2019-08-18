import argparse

from command import CommandQueue, ConvertVideoToFramesCommand, ExtractBackgroundCommand, \
    DetectObjectsCommand, SubtractBackgroundCommand, ImproveDetectionCommand, ObjectTrackingCommand,\
    CreateSummaryCommand
from observer import ProgressObserver
from video import Video, VideoWithDir


class VideoSummarizationFacade(object):
    def __init__(self, video_path, detector):
        self.video = VideoWithDir(Video(video_path))
        self.detector = detector

    def create(self):
        cmd_queue = CommandQueue()
        cmd_queue.add_command(ConvertVideoToFramesCommand(self.video))
        cmd_queue.add_command(ExtractBackgroundCommand(self.video))
        cmd_queue.add_command(SubtractBackgroundCommand(self.video))
        cmd_queue.add_command(DetectObjectsCommand(self.video, detector=self.detector))
        cmd_queue.add_command(ImproveDetectionCommand(self.video))
        cmd_queue.add_command(ObjectTrackingCommand(self.video))
        cmd_queue.add_command(CreateSummaryCommand(self.video, desired_objects=["car", "motor-bike"]))

        ProgressObserver(cmd_queue)

        cmd_queue.execute_all()


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--video", type=str, dest="video_path", required=True)
    arg_parser.add_argument("--detector", type=str, dest="detector")
    args = arg_parser.parse_args()

    summarized_video = VideoSummarizationFacade(args.video_path, args.detector)
    summarized_video.create()


main()

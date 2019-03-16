import argparse
import os
from tqdm import tqdm

from command import CommandQueue, ConvertVideoToFramesCommand, ExtractBackgroundCommand, \
    DetectObjectsCommand, SubtractBackgroundCommand, ImproveDetectionCommand, ObjectTrackingCommand,\
    CreateSummaryCommand
from detector import FasterRCNN, YOLO
from utilities.utils import sha1


class ProgressObserver:
    def __init__(self, observable):
        observable.register_observer(self)

    def notify(self, observable, *args, **kwargs):
        progress = kwargs.get("progress")
        if progress:
            with tqdm(total=100) as pbar:
                pbar.bar_format = "Task Completed: {postfix[0]}\n{l_bar}{bar}|"
                pbar.postfix = [kwargs.get('completed_command_description')]
                pbar.update(100 * float(progress))

            return 100 * float(progress)


class Video(object):
    def __init__(self, video_path):

        self.video_path = os.path.abspath(video_path)
        self.processing_path = os.path.join("processing", sha1(self.video_path))
        self.frames_path = os.path.join(self.processing_path, "frames")
        self.result_path = os.path.join(self.processing_path, "results")
        self.background_path = os.path.join(self.result_path, "background.jpeg")
        self.detection_path = os.path.join(self.result_path, "detections.tar.gz")
        self.improved_detection_path = os.path.join(self.result_path, "improved_detections.tar.gz")
        self.store_path = os.path.join(self.result_path, "object_store.pickled")
        self.store_data_path = os.path.join(self.result_path, "store")
        self.foreground_masks_path = os.path.join(self.processing_path, "foreground_masks")
        self.summarized_frames_path = os.path.join(self.processing_path, "summarized_frames")

        os.makedirs(self.frames_path, exist_ok=True)
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(self.store_data_path, exist_ok=True)
        os.makedirs(self.foreground_masks_path, exist_ok=True)
        os.makedirs(self.summarized_frames_path, exist_ok=True)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--detector", type=str, dest="detector", choices=['yolo', 'frcnn'], default='yolo')
    args = arg_parser.parse_args()

    if args.detector == "yolo":
        detector = YOLO
    else:
        detector = FasterRCNN

    video = Video("videos/test.m4v")

    cmd_queue = CommandQueue()
    cmd_queue.add_command(ConvertVideoToFramesCommand(video))
    cmd_queue.add_command(ExtractBackgroundCommand(video))
    cmd_queue.add_command(DetectObjectsCommand(video, detector=detector))
    cmd_queue.add_command(SubtractBackgroundCommand(video))
    cmd_queue.add_command(ImproveDetectionCommand(video))
    cmd_queue.add_command(ObjectTrackingCommand(video))
    cmd_queue.add_command(CreateSummaryCommand(video, desired_objects=["motorcycle", "person"]))

    ProgressObserver(cmd_queue)

    cmd_queue.execute_all()


main()

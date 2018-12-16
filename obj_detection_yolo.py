from utilities.utils import BATRPickle
from yolov3.yolov3 import Yolov3
import os


class Config(object):
    frames_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input", "videos", "frames")


obj_detector = Yolov3()
pickler = BATRPickle(out_file=os.path.join("input", "detection_result.tar.gz"))
for frame_n, frame_filename in enumerate(sorted(os.listdir(Config.frames_dir)), 1):
    frame_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "input", "videos", "frames", frame_filename)
    print(frame_filename)
    detected_objects = obj_detector.detect_image(frame_filename)
    pickler.pickle(detected_objects, f"frame{frame_n:06d}_yolo")
    if frame_n == 10:
        break
pickler.close()

import os
import time

import cv2
from yolov3.yolov3 import Yolov3

from utilities.utils import BATRPickle


def yolo_drawbbox(in_frame_folder, in_detection_result_file, out_folder=None, start=1, end=None, show_video=False):
    in_frame_folder = os.path.abspath(in_frame_folder)
    in_detection_result_file = os.path.abspath(in_detection_result_file)
    if out_folder is not None:
        out_folder = os.path.abspath(out_folder)

    pickler = BATRPickle(in_file=in_detection_result_file)

    sorted_frames_filenames = sorted(os.listdir(in_frame_folder))

    for frame_n, frame_filename in enumerate(sorted_frames_filenames, start=start):
        frame_filename = os.path.join(in_frame_folder, frame_filename)
        detected_objects = pickler.unpickle(f"frame{frame_n:06d}")
        frame = cv2.imread(frame_filename)

        for obj in detected_objects:
            Yolov3.draw_bboxes(frame, obj)

        cv2.imshow("image", frame)
        if out_folder is not None:
            cv2.imwrite(os.path.join(out_folder, f"frame{frame_n:06d}.jpg"), frame)

        if show_video:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if frame_n == end:
            break

    cv2.destroyAllWindows()


def write_text(frame, text, pos, color=(0, 255, 255)):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2, cv2.LINE_AA)


def draw_object_track(_frame, _frame_n, _obj_trails):
    for o in _obj_trails:
        if o.manipulated:
            if not hasattr(o, "last_track_drawn"):
                o.last_track_drawn = 0
                o.last_appear = time.time()

            o.last_track_drawn += 1
            if time.time() - o.last_appear < 3:
                color = [o.track_color[0] * 255, o.track_color[1] * 255, o.track_color[2] * 255]
                prev = None
                for c in o.centroids[:o.last_track_drawn]:
                    if prev:
                        cv2.line(_frame, prev, (c.x, c.y), color, 2)

                    prev = c.x, c.y
                o.last_appear = time.time()

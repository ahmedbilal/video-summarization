import operator
import os
import pickle
from enum import Enum
from abc import ABC, abstractmethod
from video import Video

import cv2
import math
import numpy as np
import GPUtil

from detector import (YOLO, DetectedObject, FasterRCNN,
                      PseudoDetector, YOLOGPU)
from utilities.background_subtractor import background_sub
from utilities.tracking import bb_intersection_over_union, intersection, track_object
from utilities.utils import BATRPickle, extract_background
from utilities.visualization import write_text


class Color(Enum):
    RED = 0, 0, 255
    GREEN = 0, 255, 0
    BLUE = 255, 0, 0
    YELLOW = 0, 255, 255
    WHITE = 255, 255, 255
    BLACK = 0, 0, 0

# Why Command Pattern?
# We want to intercept/schedule what and how different operations are executed.
# For Example,
# 1. We may want to run CPU Demanding and GPU Demanding Commands in Parallel.
# 2. We want to add gaps between executing certain commands to let our CPU or GPU to cooldown.
# 3. We want to change how certain command is executed without changing the main code
#    (driver) which add the command to CommandQueue


class Command(ABC):
    def __init__(self, obj, description=None):
        self._obj = obj
        self.description = description

    @abstractmethod
    def execute(self):
        pass


class ConvertVideoToFramesCommand(Command):
    def __init__(self, *args, **kwargs):
        kwargs["description"] = "Video To Frame Conversion"
        super().__init__(*args, **kwargs)

    def execute(self):
        if not os.listdir(self._obj.frames_path):
            # options = "-loglevel panic -start_number 0 -b:v 10000k  -vsync 0 -an -y -q:v 5"
            options = "-start_number 0 -vf fps=5 -q:v 5"
            frame_path = os.path.join(self._obj.frames_path, 'frame%06d.jpg')
            os.system(f"ffmpeg -i {self._obj.video_path} {options} {frame_path}")
        return True


class ExtractBackgroundCommand(Command):
    def __init__(self, *args, **kwargs):
        kwargs["description"] = "Background Extraction"
        super().__init__(*args, **kwargs)

    def execute(self):
        if not os.path.exists(self._obj.background_path):
            extract_background(self._obj.frames_path, self._obj.background_path)
        return True


def have_nvidia_gpu():
    try:
        g = GPUtil.getGPUs()
        if g:
            return True
        else:
            return False
    except ValueError:
        return False


def available_computing_devices():
    devices = ["CPU"]
    if have_nvidia_gpu():
        devices.append("GPU")
    return devices


class DetectorChain(object):
    next = None

    def __init__(self, video, detector, allowed_devices):
        self.video = video
        self.detector = detector
        self.allowed_device = allowed_devices

    def set_next(self, detector):
        self.next = detector
        return self.next

    def detect(self, image_filename):
        devices = available_computing_devices()
        common_devices = set(self.allowed_device).intersection(devices)
        if common_devices:
            return self.detect_image(image_filename)
        else:
            return self.next.detect(image_filename=image_filename)

    def detect_image(self, image_filename):
        # print(self.detector, "is handling.")
        return self.detector.detect(image_filename)

    def __repr__(self):
        return str(self.detector)


class DetectObjectsCommand(Command):
    def __init__(self, *args, **kwargs):
        kwargs["description"] = "Object Detection"

        self.detector = kwargs.pop("detector")

        super().__init__(*args, **kwargs)

    def execute(self):
        if self.detector not in ["yolo", "frcnn", "pseudo"]:
            self.detector = DetectorChain(self._obj, YOLOGPU(),
                                          allowed_devices=["GPU"])
            self.detector.set_next(DetectorChain(self._obj, YOLO(),
                                   allowed_devices=["CPU"]))
            self.detector.set_next(DetectorChain(self._obj, FasterRCNN(),
                                                 allowed_devices=["CPU"]))

        if not os.path.exists(self._obj.detection_path):
            if self.detector == "yolo":
                self.detector = YOLO()
            elif self.detector == "frcnn":
                self.detector = FasterRCNN()
            detect_and_save(self._obj, detector=self.detector)
        return True

    def __repr__(self):
        return self.detector


class SubtractBackgroundCommand(Command):
    def __init__(self, *args, **kwargs):
        kwargs["description"] = "Background Subtraction"
        super().__init__(*args, **kwargs)

    def execute(self):
        if not os.listdir(self._obj.foreground_masks_path):
            background_sub(in_frame_folder=self._obj.frames_path, out_frame_folder=self._obj.foreground_masks_path)
        return True


class ImproveDetectionCommand(Command):
    def __init__(self, *args, **kwargs):
        kwargs["description"] = "Improve Object Detection"
        super().__init__(*args, **kwargs)

    def execute(self):
        if not os.path.exists(self._obj.store_path) or not os.path.exists(self._obj.store_data_path):
            improved_fill_missing_detection(self._obj)
        return True


class CreateSummaryCommand(Command):
    def __init__(self, *args, **kwargs):
        kwargs["description"] = "Summary Created"
        self.desired_objects = kwargs.pop("desired_objects")
        super().__init__(*args, **kwargs)

    def execute(self):
        process_yolo_background_sub(self._obj, allowed_objects=self.desired_objects)
        return True


class ObjectTrackingCommand(Command):
    def __init__(self, *args, **kwargs):
        kwargs["description"] = "Object Tracking"
        super().__init__(*args, **kwargs)

    def execute(self):
        if not os.path.exists(self._obj.store_path) or not os.path.exists(self._obj.store_data_path):
            yolo_object_tracker(self._obj)
        return True


class Publisher(ABC):
    def __init__(self):
        self.__observers = []

    def register_observer(self, observer):
        self.__observers.append(observer)

    def notify_observers(self, *args, **kwargs):
        for observer in self.__observers:
            observer.notify(self, *args, **kwargs)


class CommandQueue(Publisher):
    def __init__(self):
        super().__init__()
        self.pending_commands = []
        self.completed_commands = []

    def add_command(self, command):
        self.pending_commands.append(command)

    def execute_all(self):
        while self.pending_commands:
            command = self.pending_commands.pop(0)
            success = command.execute()
            if success:
                self.completed_commands.append(command)
                completed_commands_len = len(self.completed_commands)
                pending_commands_len = len(self.pending_commands)
                total_commands_len = completed_commands_len + pending_commands_len
                self.notify_observers(progress=f"{completed_commands_len/total_commands_len}",
                                      completed_command_description=command.description)


def object_bbox_check(frame, obj):
    """
    Raise Exception if obj have invalid bounding box
    :param frame: Video Frame (Image)
    :param obj: Object whose bounding box we have to check
    :return: None
    """
    frame_shape_h, frame_shape_w, frame_channels = frame.shape
    assert obj.xa() < frame_shape_w or obj.xb() < frame_shape_w \
        or obj.ya() < frame_shape_h or obj.yb() < frame_shape_h, "Object Bounding Box > Frame Shape"


def is_spot_available(bbox, filled_location):
    """
    Return true if the location referred by `bbox` is available

    :param bbox: Bounding Box of an object
    :param filled_location: List of Locations (specified by bounding boxes)
                            that are already filled
    :return: bool
    """
    copy_filled_location = filled_location[:]
    bbox = [bbox.xa(), bbox.ya(), bbox.xb(), bbox.yb()]
    for loc in copy_filled_location:
        if bb_intersection_over_union(bbox, loc) > 0:
            return False

    return True


def available_loc(bboxes, filled_loc):
    """
    Return index of last bbox inside bboxes list that is available

    :param bboxes: Bounding Boxes of currently selected object
    :param filled_loc: List of Locations (specified by bounding boxes)
                       that are already filled
    :return: int (index)
    """
    last_good_bbox_i = None
    for _b, bbox in enumerate(bboxes):
        if is_spot_available(bbox, filled_loc):
            last_good_bbox_i = _b
        else:
            break
    return last_good_bbox_i


def set_tracker(obj, frame):
    obj.tracker = cv2.TrackerKCF_create()
    obj.tracker.init(frame, (obj.x, obj.y, obj.w, obj.h))


def apply_mask(image, mask, bbox, background):
    """
    :param image: Object's Image
    :param mask: Object's Mask
    :param bbox: Object's bbox
    :param background: Background Image where the object is to placed
    :return:
    """
    _forg = np.float32(image)

    _alpha = np.float32(mask) / 255

    _back = background[bbox.ya():bbox.yb(), bbox.xa():bbox.xb()]

    _forg = cv2.multiply(_alpha, _forg)
    _back = cv2.multiply(1.0 - _alpha, _back)

    output = cv2.add(_forg, _back)
    return output


def nearest_neighbor(obj, detected_objects):
    """
    Return index of object (from detected_objects list) nearest to our obj

    :param obj: Object whose nearest neighbour we want to find
    :param detected_objects: List of objects in which the neighbours of obj exists.
    :return: int, float or infinty, infinity
    """
    iou = dict()
    for _o, _obj in enumerate(detected_objects):
        _o = str(_o)
        iou[_o] = bb_intersection_over_union(obj.bbox(), _obj.bbox())

    if iou:
        nearest_obj_index, nearest_obj_value = max(iou.items(), key=operator.itemgetter(1))
        return int(nearest_obj_index), nearest_obj_value
    return math.inf, math.inf


def improved_fill_missing_detection(video: Video):
    in_detection_result_file = os.path.abspath(video.detection_path)

    pickler = BATRPickle(in_file=in_detection_result_file)

    tracking_path = os.path.join(os.path.dirname(in_detection_result_file), "improved_detections.tar.gz")
    output_pickler = BATRPickle(out_file=tracking_path)

    sorted_frames_abs_filenames = sorted([os.path.join(video.frames_path, filename) for filename in
                                          os.listdir(video.frames_path)])
    last_serial = 0
    previously_detected_objects = []
    for frame_n, frame_filename in enumerate(sorted_frames_abs_filenames, start=0):
        # print(f"Frame # {frame_n}\n{'-' * 10}")

        frame = cv2.imread(frame_filename)

        detected_objects = pickler.unpickle(f"frame{frame_n:06d}")

        # Find Out which object is Tracked and which is not
        for obj in detected_objects:
            set_tracker(obj, frame)

            nn_index, nn_distance = nearest_neighbor(obj, previously_detected_objects)
            if nn_distance > 0.5 and not math.isinf(nn_distance):
                previously_detected_objects[nn_index].tracked_in_next_frame = True
                obj.serial = previously_detected_objects[nn_index].serial
            else:
                # This is new object
                obj.serial = last_serial
                last_serial += 1

        # Seperate objects that are not tracked
        missed_obj = [obj for obj in previously_detected_objects if not hasattr(obj, "tracked_in_next_frame")]

        # Track these missed object in current frame
        for obj in missed_obj:
            nn_index, nn_distance = nearest_neighbor(obj, detected_objects)
            nn_index = nn_index

            # Double Check. May be it is already being tracked.

            if not math.isinf(nn_index):
                intersect = intersection(obj.bbox(), detected_objects[nn_index].bbox())

                if obj.w * obj.h == 0 or intersect/(obj.w * obj.h) > 0.7:
                    continue
                # cv2.rectangle(frame, obj.pt_a(), obj.pt_b(), COLOR['blue'], 6)
            #
            # if nn_distance > 0.5 and not math.isinf(nn_distance):
            #     continue

            ok, bbox = obj.tracker.update(frame)
            bbox = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # Found in current frame by tracker
            if ok:
                lost_and_found = DetectedObject(obj.type, obj.probability,
                                                _x=bbox[0], _y=bbox[1],
                                                _w=bbox[2], _h=bbox[3])
                lost_and_found.correct_bbox(frame)
                lost_and_found.tracker = obj.tracker
                lost_and_found.serial = obj.serial
                detected_objects.append(lost_and_found)

        previously_detected_objects = list(detected_objects)

        # Display Rectangle around objects and checking whether their
        # bbox are correct

        detected_objects_without_trackers = []
        for obj in detected_objects:
            cv2.rectangle(frame, obj.pt_a(), obj.pt_b(), Color.GREEN.value, 4)
            write_text(frame, f"{obj.serial}", (obj.cx(), obj.cy()))
            detected_objects_without_trackers.append(DetectedObject(obj.type, obj.probability,
                                                                    obj.x, obj.y, obj.w, obj.h))

        for obj in detected_objects_without_trackers:
            object_bbox_check(frame, obj)
        output_pickler.pickle(detected_objects_without_trackers, f"frame{frame_n:06d}")

        cv2.imwrite(f"output/frame{frame_n:06d}.jpg", frame)
        # cv2.imshow("f", frame)
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        #     return 0
        # cv2.destroyAllWindows()


def detect_and_save(video: Video, start=0, end=None, detector=None):
    """
    Create a file containing object detection results of frames inside in_frame_folder

    :param video: Object of Video class
    :param start: Starting Frame Number (default: 1)
    :param end: Ending Frame Number (default: last frame)
    :param detector: Detector to use for object detection
    """

    in_frame_folder = os.path.abspath(video.frames_path)
    out_file_name = os.path.abspath(video.detection_path)

    # print(in_frame_folder, out_file_name)
    print(detector)
    pickler = BATRPickle(out_file=out_file_name)

    sorted_frames_filenames = sorted(os.listdir(in_frame_folder))

    for frame_n, frame_filename in enumerate(sorted_frames_filenames, start=start):
        frame_filename = os.path.join(in_frame_folder, frame_filename)
        detected_objects = detector.detect(frame_filename)

        pickler.pickle(detected_objects, f"frame{frame_n:06d}")

        # print(f"Frame # {frame_n}")
        if frame_n == end:
            break

    del pickler


def process_yolo_background_sub(video: Video, allowed_objects=None):

    if allowed_objects is None:
        allowed_objects = ["car"]

    in_results_file = os.path.abspath(video.store_path)

    background = cv2.imread(video.background_path)
    background_float32 = np.float32(background)
    with open(in_results_file, "rb") as f:
        store = pickle.load(f)

    # only preserve obj which appears atleast in 20 frames
    store = [obj for obj in store if (len(obj.images) >= 20 and obj.type in allowed_objects)]

    for obj in store:
        obj.manipulated = False

    obj_trails = []

    for frame_n in range(1, 100):
        # print(f"Frame # {frame_n}")

        out_frame = np.copy(background)
        out_frame_path = os.path.join(video.summarized_frames_path, f"frame{frame_n:06d}.jpg")

        filled_locations = []
        store.sort(key=lambda x: x.manipulated, reverse=True)

        for obj_i, obj in enumerate(store):
            if len(obj.images) > 0:
                if obj.manipulated:
                    obj_image = cv2.imread(obj.images.pop(0))
                    obj_mask = cv2.imread(obj.masks.pop(0))
                    obj_bbox = obj.bboxes.pop(0)
                    obj_bbox_wth_class = [obj_bbox.xa(), obj_bbox.ya(), obj_bbox.xb(), obj_bbox.yb()]
                    filled_locations.append(obj_bbox_wth_class)
                    # if obj_image.shape != background_float32[bbox.ya():bbox.yb(), bbox.xa():bbox.xb()]
                    output = apply_mask(obj_image, obj_mask, obj_bbox, background_float32)

                    if output is not None:
                        out_frame[obj_bbox.ya():obj_bbox.yb(), obj_bbox.xa():obj_bbox.xb()] = output
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(out_frame, f"{obj.serial},{len(obj.images)}",
                                    (obj_bbox.xa() + obj_bbox.w // 2, obj_bbox.ya() + obj_bbox.h // 2),
                                    font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
                        # cv2.putText(out_frame, f"{obj.serial},{len(obj.images)}", (obj_bbox.xa() + obj_bbox.w//2,
                        #                                                            obj_bbox.ya() + obj_bbox.h//2),
                        #             font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

                else:
                    good_bbox_i = available_loc(obj.bboxes[:-5], filled_locations)
                    if good_bbox_i is not None:
                        obj.images = obj.images[good_bbox_i:]
                        obj.bboxes = obj.bboxes[good_bbox_i:]
                        obj.masks = obj.masks[good_bbox_i:]
                        obj.centroids = obj.centroids[good_bbox_i:]

                        obj_image = cv2.imread(obj.images.pop(0))
                        obj_mask = cv2.imread(obj.masks.pop(0))
                        obj_bbox = obj.bboxes.pop(0)

                        obj.manipulated = True

                        obj_trails.append(obj)

                        obj_bbox_wth_class = [obj_bbox.xa(), obj_bbox.ya(), obj_bbox.xb(), obj_bbox.yb()]
                        filled_locations.append(obj_bbox_wth_class)

                        output = apply_mask(obj_image, obj_mask, obj_bbox, background_float32)

                        if output is not None:
                            out_frame[obj_bbox.ya():obj_bbox.yb(), obj_bbox.xa():obj_bbox.xb()] = output
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(out_frame, f"{obj.serial},{len(obj.images)}, {good_bbox_i}",
                                        (obj_bbox.xa() + obj_bbox.w // 2, obj_bbox.ya() + obj_bbox.h // 2),
                                        font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

        # draw_object_track(out_frame, _frame_n=frame_n, _obj_trails=obj_trails)
        cv2.imwrite(out_frame_path, out_frame)

        # cv2.imshow("image", out_frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def yolo_object_tracker(video: Video, start=0, end=None):
    in_frame_folder = os.path.abspath(video.frames_path)
    in_foreground_mask_folder = os.path.abspath(video.foreground_masks_path)
    in_detection_result_file = os.path.abspath(video.detection_path)

    pickler = BATRPickle(in_file=in_detection_result_file)

    sorted_frames_filenames = sorted(os.listdir(in_frame_folder))

    store = []

    for frame_n, frame_filename in enumerate(sorted_frames_filenames, start=start):
        frame_file_abs_path = os.path.join(in_frame_folder, frame_filename)
        foreground_mask_abs_path = os.path.join(in_foreground_mask_folder, frame_filename)
        detected_objects = pickler.unpickle(f"frame{frame_n:06d}")
        frame = cv2.imread(frame_file_abs_path)
        mask = cv2.imread(foreground_mask_abs_path)
        # print(frame_filename, frame_file_abs_path, foreground_mask_abs_path)
        for obj in detected_objects:
            obj_image = frame[obj.ya():obj.yb(), obj.xa():obj.xb()]
            obj_mask = mask[obj.ya():obj.yb(), obj.xa():obj.xb()]

            if obj_mask.size > 0 and obj_image.size > 0:
                track_object(obj=obj, obj_mask=obj_mask, obj_image=obj_image, _store=store,
                             _frame_n=frame_n, _store_data_path=video.store_data_path)

        print(f"Frame# {frame_n}")

        if frame_n == end:
            break
    with open(video.store_path, "wb") as f:
        pickle.dump(store, f)

    cv2.destroyAllWindows()

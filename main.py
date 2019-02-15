import operator
import os
import pickle

import cv2
import math
import numpy as np
from yolov3.yolov3 import Yolov3, DetectedObject

from utilities.background_subtractor import background_sub
from utilities.tracking import bb_intersection_over_union, track_object, intersection
from utilities.utils import BATRPickle, sha1, extract_background
from utilities.visualization import write_text, draw_object_track

COLOR = {
         'red': (0, 0, 255),
         'green': (0, 255, 0),
         'blue': (255, 0, 0),
         'yellow': (0, 255, 255),
         'white': (255, 255, 255),
         'black': (0, 0, 0)
         }


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


def yolo_detector(in_frame_folder, out_file_name, start=1, end=None):
    """
    Create a file containing object detection results of frames inside in_frame_folder

    :param in_frame_folder: Path of Folder Containing Input Frames
    :param out_file_name: Path of Output File (that will contain detection results)
    :param start: Starting Frame Number (default: 1)
    :param end: Ending Frame Number (default: last frame)
    """

    in_frame_folder = os.path.abspath(in_frame_folder)
    out_file_name = os.path.abspath(out_file_name)

    obj_detector = Yolov3()
    pickler = BATRPickle(out_file=out_file_name)

    sorted_frames_filenames = sorted(os.listdir(in_frame_folder))

    for frame_n, frame_filename in enumerate(sorted_frames_filenames, start=start):
        frame_filename = os.path.join(in_frame_folder, frame_filename)

        detected_objects = obj_detector.detect_image(frame_filename)

        pickler.pickle(detected_objects, f"frame{frame_n:06d}")

        print(f"Frame # {frame_n}")
        if frame_n == end:
            break

    del pickler


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


def correct_bbox(obj, frame):
    """
    Modify obj to have bbox components that are within frame's size

    :param obj: Object whose bbox we want to correct.
    :param frame: Video Frame in which the obj appears
    :return: None
    """

    frame_shape_h, frame_shape_w, frame_channels = frame.shape

    obj.x = min(obj.x, frame_shape_w)
    obj.y = min(obj.y, frame_shape_h)
    obj.w = min(frame_shape_w - obj.x, obj.w)
    obj.h = min(frame_shape_h - obj.y, obj.h)


def add_tracker_to_yolo(in_detection_result_file):
    in_detection_result_file = os.path.abspath(in_detection_result_file)

    pickler = BATRPickle(in_file=in_detection_result_file)
    output_pickler = BATRPickle(out_file="detections_with_tracker.tar.gz")

    for frame_n in range(1, 1802):
        print("Frame #", frame_n)
        detected_objects = pickler.unpickle(f"frame{frame_n:06d}")
        detected_objects_fixed = []
        for obj in detected_objects:
            detected_objects_fixed.append(DetectedObject(obj.type, obj.probability,
                                                         obj.x, obj.y, obj.w, obj.h))
            obj.tracker = None
        output_pickler.pickle(detected_objects_fixed, f"frame{frame_n:06d}")


def improved_fill_missing_detection(in_frame_folder, in_detection_result_file):
    in_detection_result_file = os.path.abspath(in_detection_result_file)

    pickler = BATRPickle(in_file=in_detection_result_file)
    # output_pickler = BATRPickle(out_file="detections_improved.tar.gz")

    sorted_frames_abs_filenames = sorted([os.path.join(in_frame_folder, filename) for filename in
                                          os.listdir(in_frame_folder)])
    start = 1
    last_serial = 0
    previously_detected_objects = []
    for frame_n, frame_filename in enumerate(sorted_frames_abs_filenames[start:], start=1):
        print(f"Frame # {frame_n}\n{'-' * 10}")

        frame = cv2.imread(frame_filename)
        frame_shape_h, frame_shape_w, frame_channels = frame.shape

        detected_objects = pickler.unpickle(f"frame{frame_n:06d}")

        # Find Out which object is Tracked and which is not
        for obj in detected_objects:
            correct_bbox(obj, frame)
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

            # Double Check. May be it is already being tracked.
            intersect = intersection(obj.bbox(), detected_objects[nn_index].bbox())

            if intersect/(obj.w * obj.h) > 0.7:
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
                                                _x=min(bbox[0], frame_shape_w),
                                                _y=min(bbox[1], frame_shape_h),
                                                _w=min(frame_shape_w - bbox[0], bbox[2]),
                                                _h=min(frame_shape_h - bbox[1], bbox[3]))
                lost_and_found.tracker = obj.tracker
                lost_and_found.serial = obj.serial
                detected_objects.append(lost_and_found)

        previously_detected_objects = list(detected_objects)

        # Display Rectangle around objects and checking whether their
        # bbox are correct

        detected_objects_without_trackers = []
        for obj in detected_objects:
            cv2.rectangle(frame, obj.pt_a(), obj.pt_b(), COLOR['green'], 4)
            write_text(frame, f"{obj.serial}", (obj.cx(), obj.cy()))
            detected_objects_without_trackers.append(DetectedObject(obj.type, obj.probability,
                                                                    obj.x, obj.y, obj.w, obj.h))

        for obj in detected_objects_without_trackers:
            object_bbox_check(frame, obj)
        # output_pickler.pickle(detected_objects_without_trackers, f"frame{frame_n:06d}")

        cv2.imwrite(f"output/frame{frame_n:06d}.jpg", frame)
        cv2.imshow("f", frame)
        cv2.waitKey(1)
    cv2.destroyAllWindows()


def set_tracker(obj, frame):
    obj.tracker = cv2.TrackerKCF_create()
    obj.tracker.init(frame, (obj.x, obj.y, obj.w, obj.h))


def fill_missing_detection(in_frame_folder, in_detection_result_file):
    in_detection_result_file = os.path.abspath(in_detection_result_file)

    pickler = BATRPickle(in_file=in_detection_result_file)
    sorted_frames_abs_filenames = sorted([os.path.join(in_frame_folder, filename) for filename in
                                          os.listdir(in_frame_folder)])

    start = 1

    # output_pickler = BATRPickle(out_file="detections.tar.gz")

    last_frame_detected_obj = []
    last_serial = 0
    for frame_n, frame_filename in enumerate(sorted_frames_abs_filenames[start:], start=start):
        print(f"Frame # {frame_n}\n{'-' * 10}")

        frame = cv2.imread(frame_filename)

        detected_objects = pickler.unpickle(f"frame{frame_n:06d}")

        for obj in last_frame_detected_obj:
            nn_index, nn_distance = nearest_neighbor(obj, detected_objects)
            found_object = detected_objects[nn_index]

            if nn_distance > 0.5 and not math.isinf(nn_distance) and hasattr(obj, "serial"):
                set_tracker(found_object, frame)
                found_object.serial = obj.serial
                # cv2.rectangle(frame, found_object.pt_a(), found_object.pt_b(), COLOR['green'], 4)
            else:
                # Lost
                if obj.tracker is not None:
                    ok, bbox = obj.tracker.update(frame)
                    bbox = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), COLOR['yellow'], 6)

                    if ok:
                        lost_and_found = DetectedObject(obj.type, obj.probability,
                                                        bbox[0], bbox[1], bbox[2], bbox[3])
                        lost_and_found.serial = obj.serial
                        lost_and_found.tracker = obj.tracker
                        detected_objects.append(lost_and_found)

                        cv2.rectangle(frame, lost_and_found.pt_a(), lost_and_found.pt_b(), COLOR['yellow'], 6)
                else:
                    cv2.rectangle(frame, obj.pt_a(), obj.pt_b(), COLOR['white'], 6)

                    # Detected First Time
                    set_tracker(obj, frame)
                    obj.serial = last_serial
                    last_serial += 1

        last_frame_detected_obj = detected_objects.copy()

        # write_text(frame, f"Frame#{frame_n}", (100, 100))
        detected_objects_without_trackers = detected_objects.copy()
        for obj in detected_objects_without_trackers:
            object_bbox_check(frame, obj)
            write_text(frame, obj.serial, (obj.cx(), obj.cy()))

            cv2.rectangle(frame, obj.pt_a(), obj.pt_b(), COLOR['green'], 4)
        # output_pickler.pickle(detected_objects_without_trackers, f"frame{frame_n:06d}")

        cv2.imshow("f", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # del pickler


def yolo_object_tracker(in_frame_folder, in_detection_result_file, in_foreground_mask_folder, _store_path,
                        _store_data_path, start=1, end=None):
    in_frame_folder = os.path.abspath(in_frame_folder)
    in_foreground_mask_folder = os.path.abspath(in_foreground_mask_folder)
    in_detection_result_file = os.path.abspath(in_detection_result_file)

    pickler = BATRPickle(in_file=in_detection_result_file)

    sorted_frames_filenames = sorted(os.listdir(in_frame_folder))

    store = []

    for frame_n, frame_filename in enumerate(sorted_frames_filenames, start=start):
        frame_file_abs_path = os.path.join(in_frame_folder, frame_filename)
        foreground_mask_abs_path = os.path.join(in_foreground_mask_folder, frame_filename)
        detected_objects = pickler.unpickle(f"frame{frame_n:06d}")
        frame = cv2.imread(frame_file_abs_path)
        mask = cv2.imread(foreground_mask_abs_path)

        for obj in detected_objects:
            # fixing a bug where obj.type have prefix of "b'" and postfix of "'"
            # this bug comes from Yolo V3 detection layer
            obj.type = str(obj.type)[2:-1]

            obj_image = frame[obj.ya():obj.yb(), obj.xa():obj.xb()]
            obj_mask = mask[obj.ya():obj.yb(), obj.xa():obj.xb()]

            if obj_mask.size > 0 and obj_image.size > 0:
                track_object(obj=obj, obj_mask=obj_mask, obj_image=obj_image, _store=store,
                             _frame_n=frame_n, _store_data_path=_store_data_path)

        print(f"Frame# {frame_n}")

        if frame_n == end:
            break
    with open(_store_path, "wb") as f:
        pickle.dump(store, f)

    cv2.destroyAllWindows()


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
    print(_alpha.shape, _back.shape)

    _forg = cv2.multiply(_alpha, _forg)
    _back = cv2.multiply(1.0 - _alpha, _back)

    output = cv2.add(_forg, _back)
    return output


def process_yolo_background_sub(in_results_file, out_frame_folder,
                                in_background_file, allowed_objects=None):

    if allowed_objects is None:
        allowed_objects = ["car"]

    in_results_file = os.path.abspath(in_results_file)

    background = cv2.imread(in_background_file)
    background_float32 = np.float32(background)
    with open(in_results_file, "rb") as f:
        store = pickle.load(f)

    # only preserve obj which appears atleast in 20 frames
    store = [obj for obj in store if (len(obj.images) >= 20 and obj.type in allowed_objects)]

    for obj in store:
        obj.manipulated = False

    obj_trails = []

    for frame_n in range(1, 600):
        print(f"Frame # {frame_n}")

        out_frame = np.copy(background)
        out_frame_path = os.path.join(out_frame_folder, f"frame{frame_n:06d}.jpg")

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

        draw_object_track(out_frame, _frame_n=frame_n, _obj_trails=obj_trails)
        cv2.imwrite(out_frame_path, out_frame)

        # cv2.imshow("image", out_frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def create_summary(in_video, allowed_objects=None,
                   force_video_to_frames=False, force_background_extraction=False,
                   force_detection=False, force_foreground_computation=False,
                   force_object_tracking=False, force_create_summarized_video=False):
    if allowed_objects is None:
        allowed_objects = ["car"]

    in_video = os.path.abspath(in_video)
    video_hash = sha1(in_video)
    new_dir_path = os.path.join("processing", video_hash)

    # Create processing directory
    if not os.path.exists("processing"):
        print("1. Processing Directory Creation Initiated.")
        os.mkdir("processing")

    # Create video container directory
    if not os.path.exists(new_dir_path):
        print("2. Video Container Directory Creation Initiated.")
        os.mkdir(new_dir_path)

    # Create frames directory
    frames_dir_path = os.path.join(new_dir_path, "frames")
    if not os.path.exists(frames_dir_path) or force_video_to_frames:
        print("3. Video To Frames Conversion Initiated.")
        os.mkdir(frames_dir_path)
        os.system(f"ffmpeg -i {in_video} -vf fps=30 -qscale:v 5 {os.path.join(frames_dir_path, 'frame%06d.jpg')}")

    # Create Result directory
    result_dir_path = os.path.join(new_dir_path, "results")
    if not os.path.exists(result_dir_path):
        print("4. Result directory Creation Initiated.")
        os.mkdir(result_dir_path)

    # Background Extraction
    background_path = os.path.join(result_dir_path, "background.jpeg")
    if not os.path.exists(background_path) or force_background_extraction:
        print("5. Background Extraction Initiated.")
        extract_background(frames_dir_path, background_path)

    # Detect Objects using YOLO v3
    detection_results_path = os.path.join(result_dir_path, "detections.tar.gz")
    if not os.path.exists(detection_results_path) or force_detection:
        print("5. Object Detection Initiated.")
        yolo_detector(frames_dir_path, detection_results_path)

    # Create foreground mask directory
    foreground_mask_dir_path = os.path.join(new_dir_path, "forground_masks")
    if not os.path.exists(foreground_mask_dir_path) or force_foreground_computation:
        print("6. Foreground Computation Initiated.")

        if not os.path.exists(foreground_mask_dir_path):
            os.mkdir(foreground_mask_dir_path)

        # Create Foreground Masks
        background_sub(in_frame_folder=frames_dir_path,
                       out_frame_folder=foreground_mask_dir_path)

    # Track Object
    _store_path = os.path.join(result_dir_path, "object_store.pickled")
    _store_data_path = os.path.join(result_dir_path, "store")

    if not os.path.exists(_store_path) or not os.path.exists(_store_data_path) or force_object_tracking:
        print("7. Object Tracking Initiated.")
        if not os.path.exists(_store_data_path):
            os.mkdir(_store_data_path)

        yolo_object_tracker(in_frame_folder=frames_dir_path,
                            in_detection_result_file=detection_results_path,
                            in_foreground_mask_folder=foreground_mask_dir_path,
                            _store_path=_store_path,
                            _store_data_path=_store_data_path)

    # Create summarized video
    summarized_frames_path = os.path.join(new_dir_path, "summarized_frames")
    if not os.path.exists(summarized_frames_path) or force_create_summarized_video:
        print("8. Summarized Video Creation Initiated.")

        if not os.path.exists(summarized_frames_path):
            os.mkdir(summarized_frames_path)

        process_yolo_background_sub(in_results_file=_store_path,
                                    in_background_file=background_path,
                                    out_frame_folder=summarized_frames_path,
                                    allowed_objects=allowed_objects)

    # add_tracker_to_yolo(detection_results_path)
    # improved_fill_missing_detection(in_frame_folder=frames_dir_path, in_detection_result_file=detection_results_path)
    print("Summarized Video Created")


def main():
    create_summary("videos/ferozpurclip.mp4")


main()

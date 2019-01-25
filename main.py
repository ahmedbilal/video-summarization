import math
import operator
import os
import pickle
import random

import cv2
import numpy as np

from utilities.background_subtractor import background_sub
from utilities.tracking import bb_intersection_over_union, track_object,\
                               draw_object_track, random_color
from utilities.utils import BATRPickle, sha1, extract_background
from yolov3.yolov3 import Yolov3


def is_spot_available(bbox, filled_location):
    """
    Check whether the location referred by bbox is available
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


def yolo_detector(in_frame_folder, out_file_name, start=1, end=None):
    """
    Create a file containing object detection results of frames obtained using YOLO v3

    :param in_frame_folder: Path of Folder Containing Input Frames
    :param out_file_name: Path of Output File which would contain detection results
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


def nearest_neighbor_dist(obj, detected_obj):
    # print("last_frame_detected_obj", detected_obj)
    iou = dict()
    for _o, _obj in enumerate(detected_obj):
        _o = str(_o)
        iou[_o] = bb_intersection_over_union(obj.bbox(), _obj.bbox())

    if iou:
        nearest_obj_index, nearest_obj_value = max(iou.items(), key=operator.itemgetter(1))
        return nearest_obj_value
    return math.inf


def fill_missing_detection_using_features(in_frame_folder, in_detection_result_file):
    print(cv2.getVersionMajor(), cv2.getVersionMinor())
    in_detection_result_file = os.path.abspath(in_detection_result_file)

    pickler = BATRPickle(in_file=in_detection_result_file)
    sorted_frames_filenames = sorted(os.listdir(in_frame_folder))

    last_frame_detected_obj = []

    start = 154
    orb = cv2.ORB_create(nfeatures=10000, scoreType=cv2.ORB_FAST_SCORE)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    for frame_n, frame_filename in enumerate(sorted_frames_filenames[start:], start=start):
        print(f"Frame # {frame_n}")
        print("-" * 10)
        frame_file_abs_path = os.path.join(in_frame_folder, frame_filename)

        frame = cv2.imread(frame_file_abs_path)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detected_objects = pickler.unpickle(f"frame{frame_n:06d}")

        for obj in detected_objects:
            obj_img = frame_gray[obj.ya():obj.yb(), obj.xa():obj.xb()]
            kp, des = orb.detectAndCompute(obj_img, None)
            obj.keypoints = kp.copy()
            obj.descriptors = np.float32(des)
            obj.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            # if len(obj.keypoints) < 2:
            #     cv2.imshow("f", obj_img)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()

        detected_objects = [obj for obj in detected_objects if len(obj.keypoints) > 0]

        for obj in last_frame_detected_obj:
            # fixing a bug where obj.type have prefix of "b'" and postfix of "'"
            # this bug comes from Yolo V3 detection layer
            if not hasattr(obj, "interpolated"):
                obj.type = str(obj.type)[2:-1]

            if len(obj.keypoints) < 2:
                obj_img = frame_gray[obj.ya():obj.yb(), obj.xa():obj.xb()]
                cv2.imshow("im", obj_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                cv2.rectangle(frame, (obj.xa(), obj.ya()), (obj.xb(), obj.yb()), (255, 255, 255), 4)
                continue

            closest_neighbour = None

            for _obj in detected_objects:
                if len(_obj.descriptors) < 2:
                    continue

                matches = flann.knnMatch(obj.descriptors, _obj.descriptors, k=2)
                good = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good.append(m)
                if closest_neighbour is None:
                    closest_neighbour = _obj, len(good)
                else:
                    if closest_neighbour[1] < len(good):
                        closest_neighbour = _obj, len(good)
            print("Best Match", obj, closest_neighbour)
            if closest_neighbour is not None:
                cv2.rectangle(frame, (obj.xa(), obj.ya()), (obj.xb(), obj.yb()), (0, 255, 0), 4)
            else:
                cv2.rectangle(frame, (obj.xa(), obj.ya()), (obj.xb(), obj.yb()), (0, 0, 255), 4)
                cv2.putText(frame, obj.type, (obj.xa() + int(obj.w / 2), obj.ya() + int(obj.h / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow("f", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        last_frame_detected_obj = detected_objects.copy()


# def fill_missing_detection_using_features(in_frame_folder, in_foreground_mask_folder, in_detection_result_file,
#                            in_optflow_folder):
#     in_detection_result_file = os.path.abspath(in_detection_result_file)
#
#     pickler = BATRPickle(in_file=in_detection_result_file)
#     sorted_frames_filenames = sorted(os.listdir(in_frame_folder))
#
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
#
#     last_frame_detected_obj = []
#
#     start = 154
#     orb = cv2.ORB()
#
#     for frame_n, frame_filename in enumerate(sorted_frames_filenames[start:], start=start):
#         # print(frame_filename)
#         print(f"Frame # {frame_n}")
#         print("-" * 10)
#         frame_file_abs_path = os.path.join(in_frame_folder, frame_filename)
#         foreground_mask_abs_path = os.path.join(in_foreground_mask_folder, frame_filename)
#
#         frame = cv2.imread(frame_file_abs_path)
#         orb.detectAndCompute()
#         # mask = cv2.imread(foreground_mask_abs_path)
#         # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#         # _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
#
#         detected_objects = pickler.unpickle(f"frame{frame_n:06d}")
#
#         for obj in last_frame_detected_obj:
#             # fixing a bug where obj.type have prefix of "b'" and postfix of "'"
#             # this bug comes from Yolo V3 detection layer
#             if not hasattr(obj, "interpolated"):
#                 obj.type = str(obj.type)[2:-1]
#
#             if obj.type in ["motorbike", "person"]:
#                 cv2.rectangle(frame, (obj.xa(), obj.ya()), (obj.xb(), obj.yb()), (0, 0, 255), 4)
#                 cv2.putText(frame, obj.type,  (obj.xa() + int(obj.w/2), obj.ya() + int(obj.h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#                 continue
#
#             # print(f"{obj.bbox()}'s nearest neighbour is  {nearest_neighbor_dist(obj, detected_objects)} apart")
#             nearest_neighbor_distance = nearest_neighbor_dist(obj, detected_objects)
#             if nearest_neighbor_distance > 0.5:
#                 # print(":) FOUND!")
#                 if not hasattr(obj, "interpolated"):
#                     cv2.rectangle(frame, (obj.xa(), obj.ya()), (obj.xb(), obj.yb()), (0, 255, 0), 4)
#                 else:
#                     cv2.rectangle(frame, (obj.xa(), obj.ya()), (obj.xb(), obj.yb()), (0, 255, 255), 4)
#
#             else:
#                 # Orphaned Object
#                 adopted = obj
#                 opt_x = np.average(flow[adopted.xa():adopted.xb(), adopted.ya():adopted.yb(), 0])
#                 opt_y = np.average(flow[adopted.xa():adopted.xb(), adopted.ya():adopted.yb(), 1])
#
#                 if math.isnan(opt_x) or math.isnan(opt_y):
#                     opt_x = 0
#                     opt_y = 0
#                 else:
#                     opt_x = round(opt_x, 1)
#                     opt_y = round(opt_y, 1)
#
#                 print("OBJ", adopted.x, adopted.y)
#                 print("OPT", opt_x, opt_y)
#
#                 adopted.x = int(adopted.x + round(opt_x, 2))
#                 adopted.y = int(adopted.y + round(opt_y, 2))
#
#                 adopted.interpolated = True
#                 detected_objects.append(adopted)
#                 cv2.rectangle(frame, (obj.xa(), obj.ya()), (obj.xb(), obj.yb()), (0, 255, 255), 4)
#         last_frame_detected_obj = detected_objects.copy()
#         # if len(orphaned_obj):
#         #     print("Orphan ed Objects", orphaned_obj)
#         cv2.imshow("f", frame)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()


def fill_missing_detection(in_frame_folder, in_foreground_mask_folder, in_detection_result_file,
                           in_optflow_folder):
    in_detection_result_file = os.path.abspath(in_detection_result_file)

    pickler = BATRPickle(in_file=in_detection_result_file)
    sorted_frames_filenames = sorted(os.listdir(in_frame_folder))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    last_frame_detected_obj = []

    start = 154
    for frame_n, frame_filename in enumerate(sorted_frames_filenames[start:], start=start):
        # print(frame_filename)
        print(f"Frame # {frame_n}")
        print("-" * 10)
        frame_file_abs_path = os.path.join(in_frame_folder, frame_filename)
        foreground_mask_abs_path = os.path.join(in_foreground_mask_folder, frame_filename)
        optical_flow_abs_path = os.path.join(in_optflow_folder, f"frame{frame_n:06d}.flo")
        flow = np.load(optical_flow_abs_path)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1], angleInDegrees=True)

        frame = cv2.imread(frame_file_abs_path)

        mask = cv2.imread(foreground_mask_abs_path)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

        detected_objects = pickler.unpickle(f"frame{frame_n:06d}")
        orphaned_obj = []
        # print("Objects")
        # print(detected_objects)
        for obj in last_frame_detected_obj:
            # fixing a bug where obj.type have prefix of "b'" and postfix of "'"
            # this bug comes from Yolo V3 detection layer
            if not hasattr(obj, "interpolated"):
                obj.type = str(obj.type)[2:-1]

            if obj.type in ["motorbike", "person"]:
                cv2.rectangle(frame, (obj.xa(), obj.ya()), (obj.xb(), obj.yb()), (0, 0, 255), 4)
                # print("Not wanted", obj.type)
                cv2.putText(frame, obj.type,  (obj.xa() + int(obj.w/2), obj.ya() + int(obj.h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                continue
            # print(f"{obj.bbox()}'s nearest neighbour is  {nearest_neighbor_dist(obj, detected_objects)} apart")
            nearest_neighbor_distance = nearest_neighbor_dist(obj, detected_objects)
            if nearest_neighbor_distance > 0.5:
                # print(":) FOUND!")
                if not hasattr(obj, "interpolated"):
                    cv2.rectangle(frame, (obj.xa(), obj.ya()), (obj.xb(), obj.yb()), (0, 255, 0), 4)
                else:
                    cv2.rectangle(frame, (obj.xa(), obj.ya()), (obj.xb(), obj.yb()), (0, 255, 255), 4)

            else:
                # print(":( NOT FOUND!")
                orphaned_obj.append(obj)
                adopted = obj
                opt_x = np.average(flow[adopted.xa():adopted.xb(), adopted.ya():adopted.yb(), 0])
                opt_y = np.average(flow[adopted.xa():adopted.xb(), adopted.ya():adopted.yb(), 1])

                if math.isnan(opt_x) or math.isnan(opt_y):
                    opt_x = 0
                    opt_y = 0
                else:
                    opt_x = round(opt_x, 1)
                    opt_y = round(opt_y, 1)

                print("OBJ", adopted.x, adopted.y)
                print("OPT", opt_x, opt_y)

                adopted.x = int(adopted.x + round(opt_x, 2))
                adopted.y = int(adopted.y + round(opt_y, 2))

                adopted.interpolated = True
                detected_objects.append(adopted)
                cv2.rectangle(frame, (obj.xa(), obj.ya()), (obj.xb(), obj.yb()), (0, 255, 255), 4)
        last_frame_detected_obj = detected_objects.copy()
        # if len(orphaned_obj):
        #     print("Orphan ed Objects", orphaned_obj)
        cv2.imshow("f", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def yolo_object_tracker(in_frame_folder, in_detection_result_file, in_foreground_mask_folder, _store_path,
                        _store_data_path, start=1, end=None):
    in_frame_folder = os.path.abspath(in_frame_folder)
    in_foreground_mask_folder = os.path.abspath(in_foreground_mask_folder)
    in_detection_result_file = os.path.abspath(in_detection_result_file)

    pickler = BATRPickle(in_file=in_detection_result_file)

    sorted_frames_filenames = sorted(os.listdir(in_frame_folder))

    store = []

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

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

        print(frame_n)

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

    _forg = cv2.multiply(_alpha, _forg)
    _back = cv2.multiply(1.0 - _alpha, _back)

    output = cv2.add(_forg, _back)
    return output


def process_yolo_background_sub(in_results_file, out_frame_folder,
                                in_background_file, show_video=False,
                                allowed_objects=None):

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

        cv2.imshow("image", out_frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def yolo_deeplab(in_frame_folder, in_detection_result_file, in_mask_file, out_frame_folder, start=1,
                 end=None, show_video=False):
    in_frame_folder = os.path.abspath(in_frame_folder)
    out_frame_folder = os.path.abspath(out_frame_folder)

    in_mask_file = os.path.abspath(in_mask_file)
    in_detection_result_file = os.path.abspath(in_detection_result_file)

    background = cv2.imread("input/testavg.jpg")

    pickler = BATRPickle(in_file=in_detection_result_file)
    mask_pickler = BATRPickle(in_file=in_mask_file)

    sorted_frames_filenames = sorted(os.listdir(in_frame_folder))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    store = []
    # deeplabmasks.tar.gz

    for frame_n, frame_filename in enumerate(sorted_frames_filenames, start=start):
        out_frame = np.copy(background)
        frame_filename = os.path.join(in_frame_folder, frame_filename)
        detected_objects = pickler.unpickle(f"frame{frame_n:06d}")
        frame = cv2.imread(frame_filename)
        for obj in detected_objects:
            if obj.type != "b'car'":
                continue
            obj_image = frame[obj.ya:obj.yb, obj.xa:obj.xb]

            track_object(obj, obj_image, store)
            mask = mask_pickler.unpickle(f"frame{frame_n:06d}")
            mask[mask > 0] = 255

            # cv2.imshow("Mask", mask)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            cv2.imwrite("mask.jpg", mask)

            _alpha = cv2.imread("mask.jpg")
            # _alpha2 = cv2.dilate(_alpha, kernel2, iterations=10)
            _alpha = cv2.dilate(_alpha, kernel, iterations=3)
            # _alpha = cv2.morphologyEx(_alpha, cv2.MORPH_CLOSE, kernel)

            # _common = cv2.morphologyEx(_common, cv2.MORPH_CLOSE, kernel)
            #
            # cv2.imshow("image", _common)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            _forg = np.float32(obj_image)

            _back = np.float32(background[obj.ya:obj.yb, obj.xa:obj.xb])
            _alpha = np.float32(_alpha) / 255

            _forg = cv2.multiply(_alpha[obj.ya:obj.yb, obj.xa:obj.xb], _forg)
            _back = cv2.multiply(1.0 - _alpha[obj.ya:obj.yb, obj.xa:obj.xb], _back)

            output = cv2.add(_forg, _back)

            # if output is not None:
            #     cv2.imshow("image", _alpha      )
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break

            out_frame[obj.ya:obj.yb, obj.xa:obj.xb] = output

        draw_object_track(out_frame, store)

        cv2.imwrite("{}/frame{:06d}.jpg".format(out_frame_folder, frame_n), out_frame)
        print(frame_n)
        if show_video:
            cv2.imshow("image", out_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # else:
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        if frame_n == end:
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
                                    show_video=False,
                                    allowed_objects=allowed_objects)

    # fill_missing_detection_using_features(in_frame_folder=frames_dir_path,
    #                                       in_detection_result_file=detection_results_path)

    print("Summarized Video Created")


def main():
    create_summary("videos/ferozpurclip.mp4", force_create_summarized_video=True)


main()

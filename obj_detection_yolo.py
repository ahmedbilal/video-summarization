from utilities.utils import BATRPickle
from yolov3.yolov3 import Yolov3, DetectedObject

import os
import cv2
import numpy as np
import random
import operator
import time
import pickle


def obj_average_color(_image):
    avg_color_per_row = np.average(_image, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return avg_color


# class StoreObject(object):
#     def __init__(self, centroid, _image, _color, _bbox, _type):
#         self.centroids = [centroid]
#         self.track_color = random_color()
#         self.color = _color
#         self.image = _image
#         self.bbox = _bbox
#         self.type = _type
#         self.last_appear = time.time()


class Centroid(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


def random_color():
    return random.random(), random.random(), random.random()


def bb_intersection_over_union(box_a, box_b):
    # determine the (x, y)-coordinates of the intersection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # compute the area of intersection rectangle
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    # return the intersection over union value
    return iou


def track_object(obj, obj_image, _store):
    """
    Auhor: Ahmad Bilal Khalid
    Contributor: None

    Returns color for an object based on its centroid
    :type _store: list
    :type obj: DetectedObject
    """
    _bbox = [obj.xa, obj.ya, obj.xb, obj.yb]
    cx_axis = int((obj.xa + obj.xb) / 2)
    cy_axis = int((obj.ya + obj.yb) / 2)
    _centroid = Centroid(cx_axis, cy_axis)

    distances = {}

    for _o, o in enumerate(_store):
        distances[str(_o)] = bb_intersection_over_union(_bbox, o.bbox)

    nearest_obj_value = None
    if distances:
        nearest_obj_index, nearest_obj_value = max(distances.items(), key=operator.itemgetter(1))
        nearest_obj_index = int(nearest_obj_index)
        if nearest_obj_value >= 0.2 and _store[nearest_obj_index].type == obj.type and obj_image.size:
            previous_color = _store[nearest_obj_index].color

            _store[nearest_obj_index].centroids.append(_centroid)
            _store[nearest_obj_index].images.append(obj_image)
            _store[nearest_obj_index].bbox = _bbox
            _store[nearest_obj_index].last_appear = time.time()
            return previous_color, nearest_obj_index, nearest_obj_value

    obj.centroids = [_centroid]
    obj.images = [obj_image]
    obj.color = random_color()
    obj.track_color = obj.color
    obj.last_appear = time.time()
    obj.bbox = _bbox

    _store.append(obj)
    return _store[-1].color, len(_store) - 1, nearest_obj_value


def draw_object_track(_frame, _store, _obj_type=None):
    for o in _store:
        if (_obj_type and _obj_type == o.type) or not _obj_type:
            if time.time() - o.last_appear > 3:
                continue
            color = [o.track_color[0] * 255, o.track_color[1] * 255, o.track_color[2] * 255]
            prev = None
            for c in o.centroids[-30:-1]:
                if prev:
                    cv2.line(_frame, prev, (c.x, c.y), color, 2)

                prev = c.x, c.y


def yolo_drawbbox(in_frame_folder, in_detection_result_file, start=1, end=None, show_video=False):
    in_frame_folder = os.path.abspath(in_frame_folder)
    in_detection_result_file = os.path.abspath(in_detection_result_file)

    pickler = BATRPickle(in_file=in_detection_result_file)

    sorted_frames_filenames = sorted(os.listdir(in_frame_folder))

    for frame_n, frame_filename in enumerate(sorted_frames_filenames, start=start):
        frame_filename = os.path.join(in_frame_folder, frame_filename)
        detected_objects = pickler.unpickle(f"frame{frame_n:06d}")
        frame = cv2.imread(frame_filename)

        for obj in detected_objects:
            Yolov3.draw_bboxes(frame, obj)

        cv2.imshow("image", frame)

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


def yolo_background_sub(in_frame_folder, in_detection_result_file, in_foreground_mask_folder,
                        out_frame_folder, start=1, end=None, show_video=False, allowed_objects=None):

    in_frame_folder = os.path.abspath(in_frame_folder)
    out_frame_folder = os.path.abspath(out_frame_folder)
    in_foreground_mask_folder = os.path.abspath(in_foreground_mask_folder)
    print(in_foreground_mask_folder)
    in_detection_result_file = os.path.abspath(in_detection_result_file)

    background = cv2.imread("input/testavg.jpg")
    background_float32 = np.float32(background)
    pickler = BATRPickle(in_file=in_detection_result_file)

    sorted_frames_filenames = sorted(os.listdir(in_frame_folder))

    store = []

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    for frame_n, frame_filename in enumerate(sorted_frames_filenames, start=start):
        out_frame = np.copy(background)

        frame_file_abs_path = os.path.join(in_frame_folder, frame_filename)
        foreground_mask_abs_path = os.path.join(in_foreground_mask_folder, frame_filename)
        detected_objects = pickler.unpickle(f"frame{frame_n:06d}")
        frame = cv2.imread(frame_file_abs_path)
        mask = cv2.imread(foreground_mask_abs_path)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

        _alpha = np.float32(mask) / 255

        for obj in detected_objects:
            # fixing a bug where obj.type have prefix of "b'" and postfix of "'"
            # this bug comes from Yolo V3 detection layer
            obj.type = obj.type[2:-1]

            if allowed_objects is not None and obj.type not in allowed_objects:
                continue

            obj_image = frame[obj.ya:obj.yb, obj.xa:obj.xb]

            _forg = np.float32(obj_image)

            _back = background_float32[obj.ya:obj.yb, obj.xa:obj.xb]

            _forg = cv2.multiply(_alpha[obj.ya:obj.yb, obj.xa:obj.xb], _forg)
            _back = cv2.multiply(1.0 - _alpha[obj.ya:obj.yb, obj.xa:obj.xb], _back)

            output = cv2.add(_forg, _back)
            if output is not None:
                track_object(obj=obj, obj_image=output, _store=store)
                out_frame[obj.ya:obj.yb, obj.xa:obj.xb] = output
        draw_object_track(out_frame, store)
        #
        # if output is not None:
        #     cv2.imshow("image", output/255.0)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        # cv2.imwrite("{}/frame{:06d}.jpg".format(out_frame_folder, frame_n), out_frame)
        print(frame_n)
        if show_video:
            cv2.imshow("image", out_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if frame_n == end:
            break
    with open("results.pickled", "wb") as f:
        pickle.dump(store, f)

    cv2.destroyAllWindows()


def process_yolo_background_sub(in_results_file, out_frame_folder,
                                   start=1, end=None, show_video=False,
                                   allowed_objects=None):

    in_results_file = os.path.abspath(in_results_file)

    background = cv2.imread("input/testavg.jpg")

    with open(in_results_file, "rb") as f:
        store = pickle.load(f)

    # only preserve obj which appears atleast in 20 frames
    store = [obj for obj in store if len(obj.centroids) >= 20]



    for frame_n, frame_filename in enumerate(sorted_frames_filenames, start=start):
        out_frame = np.copy(background)

        frame_file_abs_path = os.path.join(in_frame_folder, frame_filename)
        foreground_mask_abs_path = os.path.join(in_foreground_mask_folder, frame_filename)
        detected_objects = pickler.unpickle(f"frame{frame_n:06d}")
        frame = cv2.imread(frame_file_abs_path)
        mask = cv2.imread(foreground_mask_abs_path)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

        _alpha = np.float32(mask) / 255

        for obj in detected_objects:
            # fixing a bug where obj.type have prefix of "b'" and postfix of "'"
            # this bug comes from Yolo V3 detection layer
            obj.type = obj.type[2:-1]

            if allowed_objects is not None and obj.type not in allowed_objects:
                continue

            obj_image = frame[obj.ya:obj.yb, obj.xa:obj.xb]

            _forg = np.float32(obj_image)

            _back = background_float32[obj.ya:obj.yb, obj.xa:obj.xb]

            _forg = cv2.multiply(_alpha[obj.ya:obj.yb, obj.xa:obj.xb], _forg)
            _back = cv2.multiply(1.0 - _alpha[obj.ya:obj.yb, obj.xa:obj.xb], _back)

            output = cv2.add(_forg, _back)
            if output is not None:
                track_object(obj=obj, obj_image=output, _store=store)
                out_frame[obj.ya:obj.yb, obj.xa:obj.xb] = output
        draw_object_track(out_frame, store)
        #
        # if output is not None:
        #     cv2.imshow("image", output/255.0)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        # cv2.imwrite("{}/frame{:06d}.jpg".format(out_frame_folder, frame_n), out_frame)
        print(frame_n)
        if show_video:
            cv2.imshow("image", out_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if frame_n == end:
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


def background_ext(in_frame_folder, out_frame_folder, start=1, end=None):
    in_frame_folder = os.path.abspath(in_frame_folder)
    out_frame_folder = os.path.abspath(out_frame_folder)

    sorted_frames_filenames = sorted(os.listdir(in_frame_folder))
    foreground_map = cv2.createBackgroundSubtractorKNN(dist2Threshold=500)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    for frame_n, frame_filename in enumerate(sorted_frames_filenames, start=start):
        print(frame_n)

        frame_filename = os.path.join(in_frame_folder, frame_filename)
        frame = cv2.imread(frame_filename)

        mask = foreground_map.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        cv2.imwrite(f"{out_frame_folder}/frame{frame_n:06d}.jpg", mask)

        if frame_n == end:
            break

    cv2.destroyAllWindows()


def background_sub(in_frame_folder, out_frame_folder, start=1, end=None, show_video=False):
    in_frame_folder = os.path.abspath(in_frame_folder)
    out_frame_folder = os.path.abspath(out_frame_folder)

    sorted_frames_filenames = sorted(os.listdir(in_frame_folder))

    frame_filename = os.path.join(in_frame_folder, sorted_frames_filenames[0])
    background = cv2.imread(frame_filename)
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    for frame_n, frame_filename in enumerate(sorted_frames_filenames, start=start + 1):
        frame_filename = os.path.join(in_frame_folder, frame_filename)
        frame = cv2.imread(frame_filename)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)

        difference = cv2.absdiff(background_gray, frame_gray)
        _, difference = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)

        ret, thresh = cv2.threshold(difference, 127, 255, 0)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im2, contours, -1, (0, 255, 0), 3)

        # cv2.imwrite(f"{out_frame_folder}/frame{frame_n:06d}.jpg", difference)
        # print(frame_n)
        if show_video:
            cv2.imshow("image", im2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # else:
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        if frame_n == end:
            break

    cv2.destroyAllWindows()


def main():
    # yolo_detector("input/videos/frames", "output/ferozpur23122018")
    # yolo_drawbbox("input/videos/frames", "output/ferozpur23122018(1).tar.gz")
    # background_ext(in_frame_folder="input/videos/frames",
    #                  out_frame_folder="output/ferozpur_bs")
    # yolo_deeplab(in_frame_folder="input/videos/frames", in_detection_result_file="output/ferozpur23122018(1).tar.gz",
    #              in_mask_file="input/deeplabmasks.tar.gz", out_frame_folder="output/frames03012019", show_video=False)

    # yolo_background_sub(in_frame_folder="input/videos/frames",
    #                     in_detection_result_file="output/ferozpur23122018(1).tar.gz",
    #                     in_foreground_mask_folder="output/ferozpur_bs",
    #                     out_frame_folder="output/frames03012019", show_video=True,
    #                     end=250, allowed_objects=["car"])

    process_yolo_background_sub(in_results_file="input/results.pickled",
                                out_frame_folder="output/frames03012019", show_video=True,
                                end=250, allowed_objects=["car"])


main()

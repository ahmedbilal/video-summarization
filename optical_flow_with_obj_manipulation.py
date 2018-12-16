# import sys
# import operator
# import cv2
# import numpy as np
# import collections
# import time
#
# from math import sqrt
# from random import randint
#
# sys.path.append("Video-Summarization")
# sys.path.append("Mask_RCNN")
#
# from utils import BATRVideoCapture, BATRPickle, BATRTar
# from mrcnn import visualize
#
# Object = collections.namedtuple("Object", ["centroids", "obj", "color"])
# Centroid = collections.namedtuple("Centroid", ["x", "y"])
#
# class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
#                'bus', 'train', 'truck', 'boat', 'traffic light',
#                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
#                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
#                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#                'kite', 'baseball bat', 'baseball glove', 'skateboard',
#                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
#                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
#                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
#                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
#                'teddy bear', 'hair drier', 'toothbrush']
#
#
# def centroid(_image):
#     return _image.shape[1] // 2, _image.shape[0] // 2  # width, height
#
#
# def euclidean_distance(x2, y2, x1, y1):
#     return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#
#
# def color_for_object(_centroid, obj, _store, _colors):
#     """
#     Auhor: Ahmad Bilal Khalid
#     Contributor: None
#
#     Returns color for an object based on its centroid
#
#     :param _centroid:
#     :param obj:
#     :param _store:
#     :param _colors:
#     :return:
#     """
#
#     distances = {}
#
#     for _o, o in enumerate(_store):
#         o_centroids = o.centroids
#         distances[str(_o)] = euclidean_distance(o_centroids[-1].x, o_centroids[-1].y, _centroid.x, _centroid.y)
#
#     if distances:
#         nearest_obj_index, nearest_obj_value = min(distances.items(), key=operator.itemgetter(1))
#         nearest_obj_index = int(nearest_obj_index)
#
#         if nearest_obj_value <= 40:
#             previous_color = _store[nearest_obj_index][-1]
#             _store[nearest_obj_index].centroids.append(_centroid)
#             # _store[nearest_obj_index].obj = obj
#             return previous_color, nearest_obj_index
#
#     new_object = Object([_centroid], obj, _colors.pop())
#     _store.append(new_object)
#     return _store[-1][-1], len(_store) - 1
#
#
# def track_for_object(frame, _store):
#     for o in _store:
#         for _c, c in enumerate(o.centroids[:-1]):
#             c_next = o.centroids[_c + 1]
#             cv2.line(frame, c, c_next, (255, 0, 0), 2)
#
#
# offset = 0
# counter = 0
# video = BATRVideoCapture("wales_shortened.mp4", offset)
# pickler = BATRPickle(in_file="frames0-500.tar.bz2")
# optical_flow_tar = BATRTar.open("optical_flow0-750.tar.xz")
# # background = cv2.imread("wales_background.jpg")
#
# store = []
# run_video = True
#
# optical_flow_unpickler = optical_flow_pickler.unpickle()
# unpickler = pickler.unpickle()
#
# for image in video.frames:
#     _filename = None
#     results = None
#
#     optical_flow_image1_horz = None
#     optical_flow_image1_vert = None
#     optical_flow_image2_horz = None
#     optical_flow_image2_vert = None
#     try:
#         _, results = next(unpickler)
#         _, optical_flow_image1_horz = next(optical_flow_unpickler)
#         _, optical_flow_image1_vert = next(optical_flow_unpickler)
#         _, optical_flow_image2_horz = next(optical_flow_unpickler)
#         _, optical_flow_image2_vert = next(optical_flow_unpickler)
#
#     except StopIteration:
#         break
#
#     r = results[0]
#     N = r['rois'].shape[0]
#     colors = visualize.random_colors(N)
#     # image = (background - image) + (image - background)
#     # out_frame = None
#     # for i in range(N):
#     #     dim = r['rois'][i]
#     #     detected_obj_image = image[dim[0]:dim[2], dim[1]:dim[3]]
#     #     cx_axis = int((r['rois'][i][1] + r['rois'][i][3]) / 2)
#     #     cy_axis = int((r['rois'][i][0] + r['rois'][i][2]) / 2)
#     #     _centroid = Centroid(cx_axis, cy_axis)
#     #
#     #     color_for_obj, obj_index = color_for_object(_centroid, detected_obj_image, store, colors)
#     #     out_frame = visualize.apply_mask(image, r['masks'][:, :, i], color_for_obj)
#     #     cv2.putText(image, "({},{},{},{})".format(cx_axis, cy_axis, class_names[r['class_ids'][i]], obj_index),
#     #                 (cx_axis, cy_axis), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
#
#     u_xy = np.hypot(optical_flow_image1_horz, optical_flow_image2_vert)
#     v_xy = np.hypot(optical_flow_image2_horz, optical_flow_image1_vert)
#
#     # The darker the more significant the motion is
#     # 0   - Black
#     # 255 - White
#
#     u_xy[u_xy > 150] = 255
#     v_xy[v_xy > 150] = 255
#
#     out_frame = u_xy + v_xy
#     out_frame[out_frame >= 300] = 255
#
#     out_frame = np.uint8(out_frame)
#     # # track_for_object(out_frame, store)
#     cv2.imshow("output", out_frame)
#
#     if not run_video:
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     else:
#         if cv2.waitKey(5) & 0xFF == ord('q'):
#             break
#
# # print(store)
#
# if run_video:
#     cv2.destroyAllWindows()


import sys
import operator
import cv2
import numpy as np
import collections
import tarfile
from vi3o import Video
from math import sqrt

sys.path.append("Mask_RCNN")

from utilities.utils import BATRPickle
from mrcnn import visualize


class Object:
    def __init__(self, centroids, _image, color, _bbox):
        self.centroids = centroids
        self.avg_color = obj_average_color(_image)
        self.color = color
        self.image = _image
        self.bbox = _bbox


Centroid = collections.namedtuple("Centroid", ["x", "y"])

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


def obj_average_color(_image):
    avg_color_per_row = np.average(_image, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return avg_color


def euclidean_distance(x2, y2, x1, y1):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def bb_intersection_over_union(boxA, boxB):

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


# def color_for_object(_centroid, _image, _store, _colors):
#     """
#     Auhor: Ahmad Bilal Khalid
#     Contributor: None
#
#     Returns color for an object based on its centroid
#
#     :param _centroid:
#     :param _image:
#     :param _store:
#     :param _colors:
#     :return:
#     """
#     # o  = object in store
#     # _o = object's index
#     distances = {}
#     color_difference = {}
#     final = {}
#
#     for _o, o in enumerate(_store):
#         distances[str(_o)] = euclidean_distance(o.centroids[-1].x, o.centroids[-1].y, _centroid.x, _centroid.y)
#         # color_difference[str(_o)] = abs(np.sum(obj_average_color(_image) - o.avg_color))
#
#     # for _i in range(len(distances.keys())):
#     #     # current_color_diff = int(color_difference[str(_o)])
#     #     final[str(_i)] = distances[str(_i)]
#     #     # final[str(_i)] = np.hypot(distances[str(_i)], current_color_diff)
#
#     # if final:
#     if distances:
#         nearest_obj_index, nearest_obj_value = min(distances.items(), key=operator.itemgetter(1))
#         nearest_obj_index = int(nearest_obj_index)
#
#         if nearest_obj_value <= 40:
#             previous_color = _store[nearest_obj_index].color
#             _store[nearest_obj_index].centroids.append(_centroid)
#             _store[nearest_obj_index].avg_color = None
#             return previous_color, nearest_obj_index
#
#     new_object = Object([_centroid], _image, _colors.pop())
#     _store.append(new_object)
#     return _store[-1].color, len(_store) - 1



def color_for_object(_centroid, _image, _store, _colors, _bbox):
    """
    Auhor: Ahmad Bilal Khalid
    Contributor: None

    Returns color for an object based on its centroid
    """

    # o  = object in store
    # _o = object's index
    distances = {}
    color_difference = {}
    final = {}

    for _o, o in enumerate(_store):
        distances[str(_o)] = bb_intersection_over_union(_bbox, o.bbox)
        # color_difference[str(_o)] = abs(np.sum(obj_average_color(_image) - o.avg_color))

    # for _i in range(len(distances.keys())):
    #     # current_color_diff = int(color_difference[str(_o)])
    #     final[str(_i)] = distances[str(_i)]
    #     # final[str(_i)] = np.hypot(distances[str(_i)], current_color_diff)

    # if final:
    # print(distances)
    nearest_obj_value = None
    if distances:
        nearest_obj_index, nearest_obj_value = max(distances.items(), key=operator.itemgetter(1))
        nearest_obj_index = int(nearest_obj_index)

        if nearest_obj_value >= 0.2:
            previous_color = _store[nearest_obj_index].color
            _store[nearest_obj_index].centroids.append(_centroid)
            _store[nearest_obj_index].avg_color = None
            _store[nearest_obj_index].bbox = _bbox
            return previous_color, nearest_obj_index, nearest_obj_value

    # print(nearest_obj_value)
    new_object = Object([_centroid], _image, _colors.pop(), _bbox)
    _store.append(new_object)
    return _store[-1].color, len(_store) - 1, nearest_obj_value


def track_for_object(_frame, _store):
    for o in _store:
        for _c, c in enumerate(o.centroids[:-1]):
            c_next = o.centroids[_c + 1]
            cv2.line(_frame, c, c_next, (255, 0, 0), 2)


# Configuration
CONFIG = dict()
CONFIG["offset"] = 0
CONFIG["end"] = 1000

CONFIG["input_video_file"] = "input/videos/wales_shortened.mp4"
CONFIG["input_mask_file"] = "input/maskrcnn0-5000.tar.gz"
CONFIG["input_opticalflow_file"] = "input/optical_flow0-750.tar.xz"

CONFIG["display_video"] = False
CONFIG["display_image"] = False

CONFIG["display_object_info"] = False

CONFIG["create_video"] = True
CONFIG["output_video_file"] = "output/wales_shortened_iou.avi"


video = Video(CONFIG["input_video_file"])
pickled_masks = BATRPickle(in_file=CONFIG["input_mask_file"])
output_video = None
if CONFIG["create_video"]:
    output_video = cv2.VideoWriter(CONFIG["output_video_file"], cv2.VideoWriter_fourcc(*"X264"),
                                   30, tuple(reversed(video[0].shape[:-1])), True)

optical_flow_tar = tarfile.open(CONFIG["input_opticalflow_file"], "r:xz")

store = []

for frame_n in range(0, min(len(video), CONFIG["end"])):
    print(frame_n)
    frame = video[frame_n + 1]

    optical_flow_member = optical_flow_tar.getmember("frame{:06d}-{:06d}.jpg".format(frame_n, frame_n + 1))
    optical_flow_extracted_member = optical_flow_tar.extractfile(optical_flow_member)
    optical_flow_array = np.fromstring(optical_flow_extracted_member.read(), np.uint8)
    optical_flow = cv2.imdecode(optical_flow_array, cv2.IMREAD_REDUCED_GRAYSCALE_8)
    optical_flow[:200, :] = 0
    ret, binary_optical_flow = cv2.threshold(optical_flow, 120, 255, cv2.THRESH_BINARY)

    results = pickled_masks.unpickle("frame_{:06d}".format(frame_n))
    #
    # cv2.imshow("image", optical_flow)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    r = results[0]
    N = r['rois'].shape[0]
    colors = visualize.random_colors(N)

    out_frame = frame
    for i in range(N):
        dim = r['rois'][i]
        detected_obj_image = frame[dim[0]:dim[2], dim[1]:dim[3]]
        cx_axis = int((r['rois'][i][1] + r['rois'][i][3]) / 2)
        cy_axis = int((r['rois'][i][0] + r['rois'][i][2]) / 2)
        _centroid = Centroid(cx_axis, cy_axis)
        bbox = [r['rois'][i][1], r['rois'][i][0], r['rois'][i][3], r['rois'][i][2]]

        color_for_obj, obj_index, near_value = color_for_object(_centroid, detected_obj_image, store, colors, bbox)

        current_obj_mask = 1 * r['masks'][:, :, i]
        value = sum(sum(binary_optical_flow[dim[0]:dim[2], dim[1]:dim[3]] * current_obj_mask[dim[0]:dim[2], dim[1]:dim[3]])) /\
                np.count_nonzero(current_obj_mask[dim[0]:dim[2], dim[1]:dim[3]] == 1)
        # cv2.imshow("meow", out_frame)
        if value > 30:
            out_frame = visualize.apply_mask(frame, r['masks'][:, :, i], color_for_obj)

            if CONFIG["display_object_info"]:
                cv2.putText(out_frame, "({},{},{},{},{})".format(cx_axis, cy_axis, class_names[r['class_ids'][i]], obj_index, value),
                    (cx_axis, cy_axis), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                # cv2.putText(out_frame,
                #             "({})".format(str(near_value)),(cx_axis, cy_axis), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    if CONFIG["create_video"] and output_video:
        output_video.write(cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB))

    if CONFIG["display_image"]:
        cv2.imshow("output", cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB))

        # cv2.imshow("image", cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB))
        # cv2.imshow("opt", binary_optical_flow)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            sys.exit(0)
        cv2.destroyAllWindows()

    if CONFIG["display_video"]:
        cv2.imshow("output", cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB))
        # cv2.imshow("opt", binary_optical_flow)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            sys.exit(0)


if CONFIG["create_video"] and output_video:
    output_video.release()

cv2.destroyAllWindows()

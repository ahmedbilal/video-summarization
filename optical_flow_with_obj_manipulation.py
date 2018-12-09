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
import time

from math import sqrt
from random import randint

sys.path.append("Video-Summarization")
sys.path.append("Mask_RCNN")

from utils import BATRVideoCapture, BATRPickle, BATRTar
from mrcnn import visualize

Object = collections.namedtuple("Object", ["centroids", "obj", "color"])
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


def centroid(_image):
    return _image.shape[1] // 2, _image.shape[0] // 2  # width, height


def euclidean_distance(x2, y2, x1, y1):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def color_for_object(_centroid, obj, _store, _colors):
    """
    Auhor: Ahmad Bilal Khalid
    Contributor: None

    Returns color for an object based on its centroid

    :param _centroid:
    :param obj:
    :param _store:
    :param _colors:
    :return:
    """

    distances = {}

    for _o, o in enumerate(_store):
        o_centroids = o.centroids
        distances[str(_o)] = euclidean_distance(o_centroids[-1].x, o_centroids[-1].y, _centroid.x, _centroid.y)

    if distances:
        nearest_obj_index, nearest_obj_value = min(distances.items(), key=operator.itemgetter(1))
        nearest_obj_index = int(nearest_obj_index)

        if nearest_obj_value <= 40:
            previous_color = _store[nearest_obj_index][-1]
            _store[nearest_obj_index].centroids.append(_centroid)
            # _store[nearest_obj_index].obj = obj
            return previous_color, nearest_obj_index

    new_object = Object([_centroid], obj, _colors.pop())
    _store.append(new_object)
    return _store[-1][-1], len(_store) - 1


def track_for_object(frame, _store):
    for o in _store:
        for _c, c in enumerate(o.centroids[:-1]):
            c_next = o.centroids[_c + 1]
            cv2.line(frame, c, c_next, (255, 0, 0), 2)


offset = 0
counter = 0

video = BATRVideoCapture("wales_shortened.mp4", offset)
background = cv2.imread("wales_background.jpg")


pickler = BATRPickle(in_file="frames0-500.tar.bz2")
unpickler = pickler.unpickle()

optical_flow_tar = BATRTar.open("optical_flow0-750.tar.xz")
optical_flow_unpickler = optical_flow_tar.allextractedfiles()
store = []
run_video = True

for image in video.frames:
    # image = (background - image) + (image - background)

    _filename = None
    results = None

    try:
        _, results = next(unpickler)
        content_of_optical_flow = np.fromstring(next(optical_flow_unpickler).read(), np.uint8)
        optical_flow = cv2.imdecode(content_of_optical_flow, cv2.IMREAD_REDUCED_GRAYSCALE_8)
    except StopIteration:
        break

    r = results[0]
    N = r['rois'].shape[0]
    colors = visualize.random_colors(N)

    out_frame = image
    m = np.zeros(image.shape[:-1])
    for i in range(N):
        # scaled_mask = 1.0 * r['masks'][:, :, i]
        # if np.count_nonzero(scaled_mask > 1):
        #     print("done")
        #     assert False
        # cv2.imshow("masl", scaled_mask)
        dim = r['rois'][i]
        detected_obj_image = image[dim[0]:dim[2], dim[1]:dim[3]]
        cx_axis = int((r['rois'][i][1] + r['rois'][i][3]) / 2)
        cy_axis = int((r['rois'][i][0] + r['rois'][i][2]) / 2)
        _centroid = Centroid(cx_axis, cy_axis)
        color_for_obj, obj_index = color_for_object(_centroid, detected_obj_image, store, colors)
        # print("Mask for ", i)
        # cv2.imshow("m*i", np.bitwise_or(cv2.cvtColor(optical_flow, cv2.COLOR_GRAY2BGR),
        #                                 cv2.cvtColor(np.uint8(255 * r['masks'][:,:,i]), cv2.COLOR_GRAY2BGR)))

        # print(obj_index, np.count_nonzero(r['masks'][:,:,i] == True))
        # cv2.imshow("opticalflow", optical_flow)
        # cv2.imshow("s", np.uint8(255 * r['masks'][:, :, i]))
        # out_frame = np.bitwise_and(optical_flow, np.uint8(1.0 * r['masks'][:, :, i]))
        #
        # value = np.sum(np.bitwise_and(optical_flow, np.uint8(1 * r['masks'][:, :, i])))
        # print(value, obj_index)
        # m = (np.bitwise_or(np.uint8(m), np.uint8(255.0 * r['masks'][:, :, i])))
        # cv2.imshow("m", np.bitwise_and(optical_flow, m))

        # time.sleep(0.1)

        # int_mask = cv2.cvtColor(np.uint8(255.0 * r['masks'][:, :, i]), cv2.COLOR_GRAY2BGR)
        # cv2.imshow("meow", int_mask)


        # scalar_product = int_mask * optical_flow
        # value = np.count_nonzero(scalar_product == 0) / (np.count_nonzero(scalar_product == 1) or 1)
        # print(obj_index, value)
        # cv2.imshow("op", optical_flow)
        # cv2.imshow("mask", int_mask)
        # value = np.sum(cv2.cvtColor(optical_flow, cv2.COLOR_BGR2GRAY) * np.uint8(r['masks'][:, :, i])) /\
        #     (detected_obj_image.shape[0] * detected_obj_image.shape[1])
        current_obj_mask = 1 * r['masks'][:, :, i]
        # value = sum(np.dot(optical_flow[dim[0]:dim[2], dim[1]:dim[3]],
        #                np.transpose(current_obj_mask[dim[0]:dim[2], dim[1]:dim[3]])))
        value = sum(sum(optical_flow[dim[0]:dim[2], dim[1]:dim[3]] * current_obj_mask[dim[0]:dim[2], dim[1]:dim[3]])) /\
                np.count_nonzero(current_obj_mask[dim[0]:dim[2], dim[1]:dim[3]] == 1)
        # value = np.dot(optical_flow, np.uint8(current_obj_mask).reshape((720, 1280, 1)))
        print(obj_index, value)
        # cv2.imshow("meow", out_frame)
        if value > 50:
            out_frame = visualize.apply_mask(image, r['masks'][:, :, i], color_for_obj)
            cv2.putText(out_frame, "({},{},{},{},{})".format(cx_axis, cy_axis, class_names[r['class_ids'][i]], obj_index, value),
                (cx_axis, cy_axis), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    # time.sleep(0.5)
    # out_frame = np.uint8(out_frame)
    # # track_for_object(out_frame, store)
    # cv2.imshow("m", np.bitwise_and(optical_flow, m))

    cv2.imshow("output", out_frame)

    if not run_video:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

# print(store)

if run_video:
    cv2.destroyAllWindows()


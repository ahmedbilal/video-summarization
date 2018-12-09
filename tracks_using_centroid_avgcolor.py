import cv2
import numpy as np
import collections
import operator

from math import sqrt
from .utilities.utils import BATRVideoCapture, BATRPickle
from video_summarization.Mask_RCNN.mrcnn import visualize


class Object:
    def __init__(self, centroids, _image, color):
        self.centroids = centroids
        self.avg_color = obj_average_color(_image)
        self.color = color


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


def color_for_object(_centroid, _image, _store, _colors):
    """
    Auhor: Ahmad Bilal Khalid
    Contributor: None

    Returns color for an object based on its centroid

    :param _centroid:
    :param _image:
    :param _store:
    :param _colors:
    :return:
    """
    # o  = object in store
    # _o = object's index
    distances = {}
    color_difference = {}
    final = {}

    for _o, o in enumerate(_store):
        distances[str(_o)] = euclidean_distance(o.centroids[-1].x, o.centroids[-1].y, _centroid.x, _centroid.y)
        color_difference[str(_o)] = abs(np.sum(obj_average_color(_image) - o.avg_color))

    for _i in range(len(distances.keys())):
        current_color_diff = int(color_difference[str(_o)])
        final[str(_i)] = np.hypot(distances[str(_i)], current_color_diff)

    if final:
        nearest_obj_index, nearest_obj_value = min(final.items(), key=operator.itemgetter(1))
        nearest_obj_index = int(nearest_obj_index)

        if nearest_obj_value < 10:
            previous_color = _store[nearest_obj_index].color
            _store[nearest_obj_index].centroids.append(_centroid)
            _store[nearest_obj_index].avg_color = obj_average_color(_image)
            return previous_color, nearest_obj_index

    new_object = Object([_centroid], _image, _colors.pop())
    _store.append(new_object)
    return _store[-1].color, len(_store) - 1


def track_for_object(frame, _store):
    for o in _store:
        for _c, c in enumerate(o.centroids[:-1]):
            c_next = o.centroids[_c+1]
            cv2.line(frame, c, c_next, (0, 0, 255), 2)


offset = 0
counter = 0
video = BATRVideoCapture("input/videos/wales_shortened.mp4", offset)
pickler = BATRPickle(in_file="input/frames0-500.tar.bz2")
unpickler = pickler.unpickle()
store = []

run_video = True

for image in video.frames:
    _filename = None
    try:
        _filename, results = next(unpickler)
    except StopIteration:
        break

    r = results[0]
    N = r['rois'].shape[0]
    colors = visualize.random_colors(N)
    out_frame = None
    for i in range(N):
        dim = r['rois'][i]
        detected_obj_image = image[dim[0]:dim[2], dim[1]:dim[3]]
        cx_axis = int((r['rois'][i][1] + r['rois'][i][3]) / 2)
        cy_axis = int((r['rois'][i][0] + r['rois'][i][2]) / 2)
        _centroid = Centroid(cx_axis, cy_axis)

        color_for_obj, obj_index = color_for_object(_centroid, detected_obj_image, store, colors)
        out_frame = visualize.apply_mask(image, r['masks'][:, :, i], color_for_obj)
        cv2.putText(image, "({},{},{},{})".format(cx_axis, cy_axis, class_names[r['class_ids'][i]], obj_index),
                    (cx_axis, cy_axis), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    track_for_object(out_frame, store)
    cv2.imshow("output", out_frame)
    if not run_video:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("-" * 12)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# print(store)

if run_video:
    cv2.destroyAllWindows()

import operator
import cv2
import collections
import sys

from math import sqrt

from utilities.utils import BATRVideoCapture, BATRPickle

sys.path.append("Mask_RCNN")
from mrcnn import visualize


class Object(object):
    def __init__(self, centroids, obj, color):
        self.centroids = centroids
        self.obj = obj
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
            previous_color = _store[nearest_obj_index].color
            _store[nearest_obj_index].centroids.append(_centroid)
            _store[nearest_obj_index].obj = obj
            return previous_color, nearest_obj_index

    new_object = Object([_centroid], obj, _colors.pop())
    _store.append(new_object)
    return _store[-1].color, len(_store) - 1


def track_for_object(frame, _store):
    for o in _store:
        for _c, c in enumerate(o.centroids[:-1]):
            c_next = o.centroids[_c+1]
            cv2.line(frame, c, c_next, (0, 255, 255), 4)


offset = 0
counter = 0
video = BATRVideoCapture("input/videos/wales_shortened.mp4", offset)
background = cv2.imread("input/wales_background.jpg")
pickler = BATRPickle(in_file="input/frames0-500.tar.bz2")
output_video = cv2.VideoWriter("output/tracks.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, video.dimensions)
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
    # image = (background - image) + (image - background)
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
    output_video.write(out_frame)
    if not run_video:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    counter += 1

    if counter == 250:
        break

# print(store)

if run_video:
    cv2.destroyAllWindows()

output_video.release()

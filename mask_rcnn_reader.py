import operator
import cv2
import collections
import sys
from vi3o import Video

from math import sqrt
from video_summarization.utilities.utils import BATRPickle

sys.path.append("Mask_RCNN")
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


def euclidean_distance(x2, y2, x1, y1):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def mask_color(_centroid, obj, _store, _colors):
    """
    Auhor: Ahmad Bilal Khalid
    Contributor: None

    Returns color for an object's mask based on its centroid

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


# Configuration
CONFIG = dict()
CONFIG["offset"] = 0
CONFIG["end"] = 5500

CONFIG["input_video_file"] = "input/videos/wales_shortened.mp4"
CONFIG["input_mask_file"] = "input/maskrcnn0-5000.tar.gz"

CONFIG["display_video"] = True
CONFIG["display_image"] = False

CONFIG["subtract_background"] = False
CONFIG["background_file"] = "input/wales_background.jpg"

CONFIG["create_masked_video"] = False
CONFIG["output_video_file"] = "output/wales_shortened_maskrcnn0-5000.mp4"
CONFIG["display_object_info"] = True


# Main Function
def main():
    video = Video(CONFIG["input_video_file"])
    output_video = None
    if CONFIG["create_masked_video"]:
        output_video = cv2.VideoWriter(CONFIG["output_video_file"], cv2.VideoWriter_fourcc(*"mp4v"),
                                       30, video[0].shape[:-1])

    background = None
    if CONFIG["subtract_background"]:
        background = cv2.imread(CONFIG["background_file"])

    pickler = BATRPickle(in_file=CONFIG["input_mask_file"])
    store = []
    for frame_n in range(CONFIG["offset"], CONFIG["end"]):
        frame = cv2.cvtColor(video[frame_n], cv2.COLOR_BGR2RGB)
        results = pickler.unpickle_file("frame_{:06d}".format(frame_n))
        r = results[0]
        n = r['rois'].shape[0]
        colors = visualize.random_colors(n)

        if CONFIG["subtract_background"]:
            frame = (background - frame) + (frame - background)

        out_frame = None
        for i in range(n):
            dim = r['rois'][i]
            detected_obj_image = frame[dim[0]:dim[2], dim[1]:dim[3]]
            cx_axis = int((r['rois'][i][1] + r['rois'][i][3]) / 2)
            cy_axis = int((r['rois'][i][0] + r['rois'][i][2]) / 2)
            _centroid = Centroid(cx_axis, cy_axis)

            _mask_color, obj_index = mask_color(_centroid, detected_obj_image, store, colors)
            out_frame = visualize.apply_mask(frame, r['masks'][:, :, i], _mask_color)

            cv2.putText(frame, "Frame # {}".format(frame_n),
                        (250, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_4)

            if CONFIG["display_object_info"]:
                cv2.putText(frame, "({},{},{},{})".format(cx_axis, cy_axis, class_names[r['class_ids'][i]], obj_index),
                            (cx_axis, cy_axis - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        if CONFIG["display_image"] or CONFIG["display_video"]:
            cv2.imshow("output", out_frame)

            if CONFIG["display_image"]:
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            elif CONFIG["display_video"]:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if CONFIG["create_masked_video"] and output_video:
            output_video.write(out_frame)

    if CONFIG["display_video"]:
        cv2.destroyAllWindows()

    if CONFIG["create_masked_video"] and output_video:
        output_video.release()


main()

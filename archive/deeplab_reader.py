
import operator
import cv2
import collections
from vi3o import Video


import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf

from math import sqrt
from utilities.utils import BATRPickle


class DeepLabOutput(object):
  def __init__(self, resized_image, segmentation_map):
    self.resized_image = resized_image
    self.segmentation_map = segmentation_map



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
CONFIG["offset"] = 1
CONFIG["end"] = 5500

CONFIG["input_video_file"] = "input/videos/wales_shortened.mp4"
CONFIG["input_mask_file"] = "input/masks(2).tar.gz"

CONFIG["display_video"] = False
CONFIG["display_image"] = True

CONFIG["subtract_background"] = False
CONFIG["background_file"] = "input/wales_background.jpg"

CONFIG["create_masked_video"] = False
CONFIG["output_video_file"] = "output/wales_shortened_maskrcnn0-5000.mp4"
CONFIG["display_object_info"] = False


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('segmentation overlay')

  unique_labels = np.unique(seg_map)
  ax = plt.subplot(grid_spec[3])
  plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.grid('off')
  plt.show()


LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)


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
        print(frame_n)

        frame = cv2.cvtColor(video[frame_n], cv2.COLOR_BGR2RGB)
        deeplab_result = pickler.unpickle("frame{:06d}".format(frame_n))

        if CONFIG["subtract_background"]:
            frame = (background - frame) + (frame - background)

        out_frame = None

        vis_segmentation(deeplab_result.resized_image, deeplab_result.segmentation_map)
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

import operator
import cv2
import collections
import sys
import numpy as np
import random
from math import sqrt
from utilities.utils import BATRPickle

sys.path.append("Mask_RCNN")
from mrcnn import visualize


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


def color_for_object(obj, _store, _colors):
    """
    Auhor: Ahmad Bilal Khalid
    Contributor: None

    Returns color for an object based on its centroid
    """
    _bbox = [obj.xa, obj.ya, obj.xb, obj.yb]
    _image = obj.image
    cx_axis = int((obj.xa + obj.xb) / 2)
    cy_axis = int((obj.ya + obj.yb) / 2)
    _centroid = Centroid(cx_axis, cy_axis)

    # o  = object in store
    # _o = object's index
    distances = {}

    for _o, o in enumerate(_store):
        distances[str(_o)] = bb_intersection_over_union(_bbox, o.bbox)

    nearest_obj_value = None
    if distances:
        nearest_obj_index, nearest_obj_value = max(distances.items(), key=operator.itemgetter(1))
        nearest_obj_index = int(nearest_obj_index)

        if nearest_obj_value >= 0.2 and _store[nearest_obj_index].type == obj.type:
            previous_color = _store[nearest_obj_index].color
            _store[nearest_obj_index].centroids.append(_centroid)
            _store[nearest_obj_index].avg_color = None
            _store[nearest_obj_index].bbox = _bbox
            return previous_color, nearest_obj_index, nearest_obj_value

    new_object = Object([_centroid], _image, _colors.pop(), _bbox, obj.type)
    _store.append(new_object)
    return _store[-1].color, len(_store) - 1, nearest_obj_value


def track_for_object(_frame, _store, _obj_type=None):
    for o in _store:
        if (_obj_type and _obj_type == o.type) or not _obj_type:
            for _c, c in enumerate(o.centroids[:-1]):
                c_next = o.centroids[_c + 1]
                color = [o.color[0] * 255, o.color[1] * 255, o.color[2] * 255]
                cv2.line(_frame, c, c_next, color, 2)


def obj_average_color(_image):
    avg_color_per_row = np.average(_image, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return avg_color


class DetectedObject(object):
    def __init__(self, _type, _probability, image, _xa, _ya, _xb, _yb, _w, _h):
        self.type = _type
        self.probability = _probability
        self.xa = _xa
        self.ya = _ya
        self.xb = _xb
        self.yb = _yb
        self.w = _w
        self.h = _h
        self.image = image


class Object:
    def __init__(self, centroids, _image, color, _bbox, _type):
        self.centroids = centroids
        self.avg_color = obj_average_color(_image)
        self.color = color
        self.image = _image
        self.bbox = _bbox
        self.type = _type


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


class Config(object):
    offset = 1
    end = 1802

    input_video_file = "input/videos/ferozpurclip.mp4"
    input_mask_file = "input/maskrcnnasif.tar.xz"

    display_video = False
    display_image = False

    subtract_background = True
    background_file = "input/testavg.jpg"

    create_masked_video = True
    output_video_file = "output/ferozpur_back_subtract.avi"
    display_object_info = False


# backSub = cv2.createBackgroundSubtractorKNN(dist2Threshold=3000, detectShadows=True)

# Main Function
def main():
    # video = Video(Config.input_video_file)
    output_video = None
    if Config.create_masked_video:
        # output_video = cv2.VideoWriter(CONFIG["output_video_file"], cv2.VideoWriter_fourcc(*"mp4v"),
        #                                30, frame.shape[:-1])
        pass

    background = cv2.imread(Config.background_file)

    if Config.subtract_background:
        background = cv2.imread(Config.background_file)

    pickler = BATRPickle(in_file=Config.input_mask_file)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    for frame_n in range(Config.offset, Config.end):
        store = []
        out_frame = np.copy(background)
        print(frame_n)
        frame = cv2.imread(f"input/videos/frames/frame{frame_n:06d}.jpg")

        results = pickler.unpickle("frame{:06d}".format(frame_n))
        r = results[0]
        n = r['rois'].shape[0]
        colors = visualize.random_colors(n)


        if Config.subtract_background:
            frame = (background - frame) + (frame - background)

        for i in range(n):
            obj = DetectedObject(_type=class_names[r['class_ids'][i]],
                                 _probability=1,
                                 image=frame[r['rois'][i][0]:r['rois'][i][2], r['rois'][i][1]:r['rois'][i][3]],
                                 _xa=r['rois'][i][1],
                                 _ya=r['rois'][i][0],
                                 _xb=r['rois'][i][3],
                                 _yb=r['rois'][i][2],
                                 _w=abs(r['rois'][i][1] - r['rois'][i][3]),
                                 _h=abs(r['rois'][i][0] - r['rois'][i][2]))

            cx_axis = int((obj.xa + obj.xb) / 2)
            cy_axis = int((obj.ya + obj.yb) / 2)
            mask = np.float32(255 * r['masks'][:, :, i])

            # color_for_obj, obj_index, near_value = color_for_object(obj, store, colors)

            cv2.imwrite("mask.jpg", mask)
            # mask = None
            _alpha = cv2.imread("mask.jpg")
            # cv2.imshow("cc", mask)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            _forg = np.float32(obj.image)
            _back = np.float32(background[obj.ya:obj.yb, obj.xa:obj.xb])
            _alpha = np.float32(_alpha) / 255

            _forg = cv2.multiply(_alpha[obj.ya:obj.yb, obj.xa:obj.xb], _forg)
            _back = cv2.multiply(1.0 - _alpha[obj.ya:obj.yb, obj.xa:obj.xb], _back)

            output = cv2.add(_forg, _back)

            _forg = cv2.dilate(_forg, kernel, iterations=3)
            # cv2.imshow("cc", _forg)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            out_frame = visualize.apply_mask(frame, r['masks'][:, :, i], random.choice(colors))

            # if obj.type == "car":
            #     out_frame = visualize.apply_mask(frame, r['masks'][:, :, i], random.choice(colors))
            #     # out_frame[obj.ya:obj.yb, obj.xa:obj.xb] = output
            #
            #     # cv2.putText(frame, "Frame # {}".format(frame_n),
            #     #             (250, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_4)
            #
            #     if Config.display_object_info:
            #         cv2.putText(frame, "({},{},{})".format(cx_axis, cy_axis, obj.type),
            #                     (cx_axis, cy_axis + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        # fgMask = backSub.apply(background_copy)
        # background_copy = frame
        track_for_object(out_frame, store, "car")
        if Config.display_image or Config.display_video:
            cv2.imshow("output", out_frame)
            # cv2.imshow("output", fgMa sk)

            if Config.display_image:
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            elif Config.display_video:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        #
        if Config.create_masked_video:
            # frame = cv2.cvtColor(background_copy, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"output/ferozpur10012019_maskrcnn/frame{frame_n:06d}.jpg", out_frame)
            # output_video.write(np.uint8(out_frame))

    if Config.display_video:
        cv2.destroyAllWindows()

    if Config.create_masked_video and output_video:
        output_video.release()


main()

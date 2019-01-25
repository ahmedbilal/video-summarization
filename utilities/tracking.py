import random
import time
import os
import cv2
import operator


class StoredObject(object):
    last_serial = 0

    def __init__(self, centroid, _image, _color, _bbox, _type, _mask, _frame_n):
        self.centroids = [centroid]
        self.bboxes = [_bbox]
        self.track_color = random_color()
        self.color = _color
        self.images = [_image]
        self.frame_n = [_frame_n]
        if _image is None:
            self.images = _image

        self.type = _type
        self.last_appear = time.time()

        self.masks = [_mask]
        if _mask is None:
            self.masks = _mask

        self.serial = StoredObject.last_serial

        StoredObject.last_serial += 1

    def __repr__(self):
        return f"{self.type} - id: {self.serial}"


class Centroid(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __repr__(self):
        return f"({self.x}, {self.y})"


class BBox(object):
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def xa(self):
        return self.x

    def ya(self):
        return self.y

    def xb(self):
        return self.x + self.w

    def yb(self):
        return self.y + self.h

    def __repr__(self):
        return f"({self.x}, {self.y}, {self.w}, {self.h})"


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


def track_object(obj, obj_mask, obj_image, _store, _frame_n, _store_data_path):
    """
    Auhor: Ahmad Bilal Khalid
    Contributor: None

    Returns color for an object based on its centroid
    :param obj_mask:
    :param obj_image:
    :type _store: list
    :type obj: DetectedObject
    """

    _bbox = [obj.xa(), obj.ya(), obj.xb(), obj.yb()]
    cx_axis = int((obj.xa() + obj.xb()) / 2)
    cy_axis = int((obj.ya() + obj.yb()) / 2)
    _centroid = Centroid(cx_axis, cy_axis)

    distances = {}

    for _o, o in enumerate(_store):
        o_bbox = o.bboxes[-1]
        o_bbox = o_bbox.xa(), o_bbox.ya(), o_bbox.xb(), o_bbox.yb()
        distances[str(_o)] = bb_intersection_over_union(_bbox, o_bbox)

    _bbox = BBox(obj.x, obj.y, obj.w, obj.h)
    nearest_obj_value = None
    if distances:
        nearest_obj_index, nearest_obj_value = max(distances.items(), key=operator.itemgetter(1))
        nearest_obj_index = int(nearest_obj_index)
        previous_color = _store[nearest_obj_index].color

        if nearest_obj_value >= 0.5 and _store[nearest_obj_index].type == obj.type \
                and obj_image.size and time.time() - _store[nearest_obj_index].last_appear < 3:

            found_obj = _store[nearest_obj_index]
            found_obj.centroids.append(_centroid)
            found_obj.last_appear = time.time()

            _object_image_filename = f"object{found_obj.serial}_image{len(found_obj.images)}.jpeg"
            _object_mask_filename = f"object{found_obj.serial}_mask{len(found_obj.images)}.jpeg"

            _image_path = os.path.join(_store_data_path, _object_image_filename)
            _mask_path = os.path.join(_store_data_path, _object_mask_filename)

            cv2.imwrite(_image_path, obj_image)
            cv2.imwrite(_mask_path, obj_mask)

            _store[nearest_obj_index].images.append(_image_path)
            _store[nearest_obj_index].bboxes.append(_bbox)
            _store[nearest_obj_index].masks.append(_mask_path)
            _store[nearest_obj_index].last_appear = time.time()
            _store[nearest_obj_index].frame_n.append(_frame_n)

            return previous_color, nearest_obj_index, nearest_obj_value

    stored_obj = StoredObject(centroid=_centroid, _image=None,
                              _color=random_color(), _bbox=_bbox,
                              _type=obj.type, _mask=None, _frame_n=_frame_n)
    _store.append(stored_obj)

    image_path = os.path.join(_store_data_path, f"object{_store[-1].serial}_image0.jpeg")
    mask_path = os.path.join(_store_data_path, f"object{_store[-1].serial}_mask0.jpeg")
    cv2.imwrite(image_path, obj_image)
    cv2.imwrite(mask_path, obj_mask)

    _store[-1].images = [image_path]
    _store[-1].masks = [mask_path]

    return _store[-1].color, len(_store) - 1, nearest_obj_value


def draw_object_track(_frame, _frame_n, _obj_trails):
    for o in _obj_trails:
        if o.manipulated:
            if not hasattr(o, "last_track_drawn"):
                o.last_track_drawn = 0
                o.last_appear = time.time()

            o.last_track_drawn += 1
            if time.time() - o.last_appear < 3:
                print(time.time() - o.last_appear)
                color = [o.track_color[0] * 255, o.track_color[1] * 255, o.track_color[2] * 255]
                prev = None
                for c in o.centroids[:o.last_track_drawn]:
                    if prev:
                        cv2.line(_frame, prev, (c.x, c.y), color, 2)

                    prev = c.x, c.y
                o.last_appear = time.time()

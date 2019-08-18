import os
from abc import ABC, abstractmethod

import cv2
import numpy as np
from yolov3.yolov3 import Yolov3


class DetectedObject(object):
    def __init__(self, _type, _probability, _x, _y, _w, _h):
        self.type = _type
        self.probability = _probability
        self.tracker = None
        self.x = _x
        self.y = _y
        self.w = _w
        self.h = _h

    def cx(self):
        return int((self.xa() + self.xb()) / 2)

    def cy(self):
        return int((self.ya() + self.yb()) / 2)

    def xa(self):
        return self.x

    def ya(self):
        return self.y

    def xb(self):
        return self.x + self.w

    def yb(self):
        return self.y + self.h

    def pt_a(self):
        return self.xa(), self.ya()

    def pt_b(self):
        return self.xb(), self.yb()

    def bbox(self):
        return self.xa(), self.ya(), self.xb(), self.yb()

    def correct_bbox(self, frame):
        """
        Modify obj to have bbox components that are within frame's size

        :param frame: Video Frame in which the obj appears
        :return: None
        """

        frame_shape_h, frame_shape_w, frame_channels = frame.shape

        self.x = min(self.x, frame_shape_w)
        self.y = min(self.y, frame_shape_h)
        self.w = min(frame_shape_w - self.x, self.w)
        self.h = min(frame_shape_h - self.y, self.h)


class Detector(ABC):
    @abstractmethod
    def detect(self, image_filename):
        pass

    def detect_and_visualize(self, image_filename):
        image = cv2.imread(image_filename)

        detected_objects = self.detect(image_filename)
        for detected_object in detected_objects:
            print(detected_object.x, detected_object.y, detected_object.w, detected_object.h,
                  detected_object.type)
            cv2.rectangle(image, (detected_object.x, detected_object.y),
                          (detected_object.x + detected_object.w,
                           detected_object.h + detected_object.y), (23, 230, 210), thickness=1)
            cv2.putText(image, detected_object.type, (detected_object.x, detected_object.y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __repr__(self):
        raise NotImplementedError()


class YOLOGPU(Detector):
    def __init__(self):
        self.yolov3 = Yolov3()

    def detect(self, image_filename):
        r = self.yolov3.detect(self.yolov3.net, self.yolov3.meta,
                               image_filename.encode("ascii"))
        detection = []
        for detected_obj in r:
            _type = detected_obj[0]

            _cx = int(detected_obj[2][0])
            _cy = int(detected_obj[2][1])
            _w = int(detected_obj[2][2])
            _h = int(detected_obj[2][3])
            _x = int(_cx - _w / 2)
            _y = int(_cy - _h / 2)

            _obj = DetectedObject(_type=_type, _probability=detected_obj[1],
                                  _x=_x, _y=_y, _w=_w, _h=_h)

            detection.append(_obj)
        return detection

    def __repr__(self):
        return "YOLO GPU"


class FasterRCNN(Detector):
    # https://heartbeat.fritz.ai/real-time-object-detection-on-raspberry-pi-using-opencv-dnn-98827255fa60
    def __init__(self):
        # Loading model
        self.model = cv2.dnn.readNetFromTensorflow('models/faster-rcnn/frozen_inference_graph.pb',
                                                   'models/faster-rcnn/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')
        self.classNames = {0: 'background',
                           1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
                           7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
                           13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
                           18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
                           24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
                           32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
                           37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
                           41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
                           46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
                           51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
                           56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                           61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
                           67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
                           75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
                           80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
                           86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

    def detect(self, image_filename):
        image = cv2.imread(image_filename)

        image_height, image_width, _ = image.shape

        self.model.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True))
        output = self.model.forward()

        detected_objects = []
        for detection in output[0, 0, :, :]:
            # 1 -> Class ID, Confidence, X, Y, W, H
            confidence = detection[2]
            if confidence > .5:
                class_id = detection[1]
                class_name = self.classNames.get(class_id)
                box_x = int(detection[3] * image_width)
                box_y = int(detection[4] * image_height)
                box_width = int(detection[5] * image_width) - box_x
                box_height = int(detection[6] * image_height) - box_y
                detected_object = DetectedObject(class_name, confidence,
                                                 box_x, box_y, box_width, box_height)
                detected_object.correct_bbox(image)
                detected_objects.append(detected_object)

        return detected_objects

    def __repr__(self):
        return "FRCNN"


class YOLO(Detector):
    # https://blogs.oracle.com/meena/object-detection-using-opencv-yolo
    def __init__(self):
        self.CONFIG = 'models/yolo/yolov3.cfg'
        self.CLASSES = 'models/yolo/yolov3.txt'
        self.WEIGHTS = 'models/yolo/yolov3.weights'

        self.classes = None
        with open(self.CLASSES, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.scale = 0.00392
        self.conf_threshold = 0.5
        self.nms_threshold = 0.4
        self.net = None

    def detect(self, image_filename):
        image = cv2.imread(image_filename)

        height, width, _ = image.shape

        # read pre-trained model and config file
        self.net = cv2.dnn.readNet(self.WEIGHTS, self.CONFIG)

        # create input blob
        blob = cv2.dnn.blobFromImage(image, self.scale, (416, 416), (0, 0, 0), True, crop=False)

        # set input blob for the network
        self.net.setInput(blob)

        # run inference through the network
        # and gather predictions from output layers
        outs = self.net.forward(self.__get_output_layers())

        # initialization
        class_ids = []
        confidences = []
        boxes = []
        # for each detetion from each output layer
        # get the confidence, class id, bounding box params
        # and ignore weak detections (confidence < 0.5)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # apply non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        detected_objects = []
        # go through the detections remaining
        # after nms and draw bounding box
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = int(box[0])
            y = int(box[1])
            w = int(box[2])
            h = int(box[3])
            detected_object = DetectedObject(self.classes[class_ids[i]], confidences[i],
                                             x, y, w, h)
            detected_object.correct_bbox(image)
            detected_objects.append(detected_object)

        return detected_objects

    def __get_output_layers(self):
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        return output_layers

    def __repr__(self):
        return "YOLO"


class PseudoDetector(Detector):
    def __init__(self, annotation_file):
        with open(annotation_file, "r") as f:
            self.detections = {}
            for line in f.readlines():
                line = line.strip().split(",")

                frame = line[0]
                x = int(line[1])
                y = int(line[2])
                w = int(line[3]) - x
                h = int(line[4]) - y
                label = line[5]

                if frame in self.detections.keys():
                    self.detections[frame].append(DetectedObject(label, 1, x, y, w, h))
                else:
                    self.detections[frame] = [DetectedObject(label, 1, x, y, w, h)]

    def detect(self, image_filename):
        detection = self.detections.get(os.path.basename(image_filename), [])
        return detection

    def __repr__(self):
        return "PseudoDetector"

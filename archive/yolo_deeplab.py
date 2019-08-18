# Note: It is Broken and need to be fix before running
#       however, as it is not being used so it will not
#       be fixed.

import os
import cv2
import numpy as np

from utilities.utils import BATRPickle
from utilities.tracking import track_object, draw_object_track


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
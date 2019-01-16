import os
import pickle

import cv2
import numpy as np
from utilities.background_subtractor import background_sub
from utilities.tracking import bb_intersection_over_union, track_object, draw_object_track
from utilities.utils import BATRPickle, sha1, extract_background
from yolov3.yolov3 import Yolov3


def is_spot_available(bbox, filled_location):
    """
    Check whether the location referred by bbox is available
    :param bbox: Bounding Box of an object
    :param filled_location: List of Locations (specified by bounding boxes)
                            that are already filled
    :return: bool
    """
    copy_filled_location = filled_location[:]
    bbox = [bbox.xa(), bbox.ya(), bbox.xb(), bbox.yb()]
    for loc in copy_filled_location:
        if bb_intersection_over_union(bbox, loc) > 0:
            return False

    return True


def available_loc(bboxes, filled_loc):
    """

    :param bboxes: Bounding Boxes of currently selected object
    :param filled_loc: List of Locations (specified by bounding boxes)
                       that are already filled
    :return: int (index)
    """
    last_good_bbox_i = None
    for _b, bbox in enumerate(bboxes):
        if is_spot_available(bbox, filled_loc):
            last_good_bbox_i = _b
        else:
            break
    return last_good_bbox_i


def yolo_drawbbox(in_frame_folder, in_detection_result_file, out_folder=None, start=1, end=None, show_video=False):
    in_frame_folder = os.path.abspath(in_frame_folder)
    in_detection_result_file = os.path.abspath(in_detection_result_file)
    if out_folder is not None:
        out_folder = os.path.abspath(out_folder)

    pickler = BATRPickle(in_file=in_detection_result_file)

    sorted_frames_filenames = sorted(os.listdir(in_frame_folder))

    for frame_n, frame_filename in enumerate(sorted_frames_filenames, start=start):
        frame_filename = os.path.join(in_frame_folder, frame_filename)
        detected_objects = pickler.unpickle(f"frame{frame_n:06d}")
        frame = cv2.imread(frame_filename)

        for obj in detected_objects:
            Yolov3.draw_bboxes(frame, obj)

        cv2.imshow("image", frame)
        if out_folder is not None:
            cv2.imwrite(os.path.join(out_folder, f"frame{frame_n:06d}.jpg"), frame)

        if show_video:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if frame_n == end:
            break

    cv2.destroyAllWindows()


def yolo_detector(in_frame_folder, out_file_name, start=1, end=None):
    """
    Create a file containing object detection results of frames obtained using YOLO v3

    :param in_frame_folder: Path of Folder Containing Input Frames
    :param out_file_name: Path of Output File which would contain detection results
    :param start: Starting Frame Number (default: 1)
    :param end: Ending Frame Number (default: last frame)
    """

    in_frame_folder = os.path.abspath(in_frame_folder)
    out_file_name = os.path.abspath(out_file_name)

    obj_detector = Yolov3()
    pickler = BATRPickle(out_file=out_file_name)

    sorted_frames_filenames = sorted(os.listdir(in_frame_folder))

    for frame_n, frame_filename in enumerate(sorted_frames_filenames, start=start):
        frame_filename = os.path.join(in_frame_folder, frame_filename)

        detected_objects = obj_detector.detect_image(frame_filename)

        pickler.pickle(detected_objects, f"frame{frame_n:06d}")

        print(f"Frame # {frame_n}")
        if frame_n == end:
            break

    del pickler


def yolo_background_sub(in_frame_folder, in_detection_result_file, in_foreground_mask_folder,
                        out_frame_folder, start=1, end=None, show_video=False, allowed_objects=None):
    in_frame_folder = os.path.abspath(in_frame_folder)
    out_frame_folder = os.path.abspath(out_frame_folder)
    in_foreground_mask_folder = os.path.abspath(in_foreground_mask_folder)
    in_detection_result_file = os.path.abspath(in_detection_result_file)

    background = cv2.imread("input/testavg.jpg")
    background_float32 = np.float32(background)
    pickler = BATRPickle(in_file=in_detection_result_file)

    sorted_frames_filenames = sorted(os.listdir(in_frame_folder))

    store = []

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    for frame_n, frame_filename in enumerate(sorted_frames_filenames, start=start):
        out_frame = np.copy(background)

        frame_file_abs_path = os.path.join(in_frame_folder, frame_filename)
        foreground_mask_abs_path = os.path.join(in_foreground_mask_folder, frame_filename)
        detected_objects = pickler.unpickle(f"frame{frame_n:06d}")
        frame = cv2.imread(frame_file_abs_path)
        mask = cv2.imread(foreground_mask_abs_path)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

        _alpha = np.float32(mask) / 255

        for obj in detected_objects:
            # fixing a bug where obj.type have prefix of "b'" and postfix ofq "'"
            # this bug comes from Yolo V3 detection layer
            obj.type = str(obj.type)[2:-1]
            if allowed_objects is not None and obj.type not in allowed_objects:
                continue

            obj_image = frame[obj.ya():obj.yb(), obj.xa():obj.xb()]

            _forg = np.float32(obj_image)

            _back = background_float32[obj.ya():obj.yb(), obj.xa():obj.xb()]

            _forg = cv2.multiply(_alpha[obj.ya():obj.yb(), obj.xa():obj.xb()], _forg)
            _back = cv2.multiply(1.0 - _alpha[obj.ya():obj.yb(), obj.xa():obj.xb()], _back)

            output = cv2.add(_forg, _back)
            if output is not None:
                track_object(obj=obj, obj_mask=mask, obj_image=output, _store=store)
                out_frame[obj.ya():obj.yb(), obj.xa():obj.xb()] = output
        draw_object_track(out_frame, store)
        #
        # if output is not None:
        #     cv2.imshow("image", output/255.0)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        # cv2.imwrite("{}/frame{:06d}.jpg".format(out_frame_folder, frame_n), out_frame)
        print(frame_n)
        if show_video:
            cv2.imshow("image", out_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if frame_n == end:
            break
    with open("results.pickled", "wb") as f:
        pickle.dump(store, f)

    cv2.destroyAllWindows()


def yolo_object_tracker(in_frame_folder, in_detection_result_file, in_foreground_mask_folder, _store_path,
                        _store_data_path, start=1, end=None, allowed_objects=None):
    in_frame_folder = os.path.abspath(in_frame_folder)
    in_foreground_mask_folder = os.path.abspath(in_foreground_mask_folder)
    in_detection_result_file = os.path.abspath(in_detection_result_file)

    pickler = BATRPickle(in_file=in_detection_result_file)

    sorted_frames_filenames = sorted(os.listdir(in_frame_folder))

    store = []

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    for frame_n, frame_filename in enumerate(sorted_frames_filenames, start=start):
        frame_file_abs_path = os.path.join(in_frame_folder, frame_filename)
        foreground_mask_abs_path = os.path.join(in_foreground_mask_folder, frame_filename)
        detected_objects = pickler.unpickle(f"frame{frame_n:06d}")
        frame = cv2.imread(frame_file_abs_path)
        mask = cv2.imread(foreground_mask_abs_path)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

        for obj in detected_objects:
            # fixing a bug where obj.type have prefix of "b'" and postfix ofq "'"
            # this bug comes from Yolo V3 detection layer
            obj.type = str(obj.type)[2:-1]
            if allowed_objects is not None and obj.type not in allowed_objects:
                continue

            obj_image = frame[obj.ya():obj.yb(), obj.xa():obj.xb()]
            obj_mask = mask[obj.ya():obj.yb(), obj.xa():obj.xb()]

            if obj_mask.size > 0 and obj_image.size > 0:
                track_object(obj=obj, obj_mask=obj_mask, obj_image=obj_image, _store=store,
                             _frame_n=frame_n, _store_data_path=_store_data_path)

        print(frame_n)

        if frame_n == end:
            break
    with open(_store_path, "wb") as f:
        pickle.dump(store, f)

    cv2.destroyAllWindows()


def process_yolo_background_sub(in_results_file, out_frame_folder,
                                in_background_file, start=1,
                                end=None, show_video=False,
                                allowed_objects=None):
    in_results_file = os.path.abspath(in_results_file)

    background = cv2.imread(in_background_file)
    background_float32 = np.float32(background)
    with open(in_results_file, "rb") as f:
        store = pickle.load(f)

    # print("len(store)", len(store))

    # only preserve obj which appears atleast in 20 frames
    store = [obj for obj in store if len(obj.images) >= 20]
    # print("len(store)", len(store))

    for obj in store:
        # n = -3
        # obj.images = obj.images[:n]
        # obj.bboxes = obj.bboxes[:n]
        # obj.masks = obj.masks[:n]
        print(os.path.abspath(obj.images[len(obj.images) // 2]))
        obj.manipulated = False

    for frame_n in range(600):
        print(f"Frame # {frame_n}")
        out_frame = np.copy(background)
        filled_locations = []
        store.sort(key=lambda x: x.manipulated, reverse=True)

        for obj_i, obj in enumerate(store):
            # obj.bboxes = obj.bboxes[-10:]
            # obj.masks = obj.masks[-10:]
            # obj.images = obj.images[-10:]

            if len(obj.images) == 0:
                continue

            if obj.manipulated:
                # if not is_spot_available(obj.bboxes[0], filled_locations):
                #     continue
                obj_image = cv2.imread(obj.images.pop(0))
                obj_mask = cv2.imread(obj.masks.pop(0))
                obj_bbox = obj.bboxes.pop(0)
                obj_bbox_wth_class = [obj_bbox.xa(), obj_bbox.ya(), obj_bbox.xb(), obj_bbox.yb()]
                filled_locations.append(obj_bbox_wth_class)

                # cv2.imwrite(f"output/talha/mask{frame_ n},{obj_i}.jpg", obj_mask)
                # cv2.imwrite(f"output/talha/image{frame_n},{obj_i}.jpg", obj_image)

                _forg = np.float32(obj_image)

                _alpha = np.float32(obj_mask) / 255

                _back = background_float32[obj_bbox.ya():obj_bbox.yb(), obj_bbox.xa():obj_bbox.xb()]

                _forg = cv2.multiply(_alpha, _forg)
                _back = cv2.multiply(1.0 - _alpha, _back)

                output = cv2.add(_forg, _back)
                if output is not None:
                    out_frame[obj_bbox.ya():obj_bbox.yb(), obj_bbox.xa():obj_bbox.xb()] = output
                    # font = cv2.FONT_HERSHEY_SIMPLEX
                    # cv2.putText(out_frame, f"{obj.serial},{len(obj.images)}", (obj_bbox.xa() + obj_bbox.w//2,
                    #                                                            obj_bbox.ya() + obj_bbox.h//2),
                    #             font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

            else:
                good_bbox_i = available_loc(obj.bboxes[:-5], filled_locations)
                if good_bbox_i is not None:
                    obj.images = obj.images[good_bbox_i:]
                    obj.bboxes = obj.bboxes[good_bbox_i:]
                    obj.masks = obj.masks[good_bbox_i:]

                    obj_image = cv2.imread(obj.images.pop(0))
                    obj_mask = cv2.imread(obj.masks.pop(0))
                    obj_bbox = obj.bboxes.pop(0)

                    obj.manipulated = True

                    obj_bbox_wth_class = [obj_bbox.xa(), obj_bbox.ya(), obj_bbox.xb(), obj_bbox.yb()]
                    filled_locations.append(obj_bbox_wth_class)

                    # cv2.imwrite(f"output/talha/mask{frame_ n},{obj_i}.jpg", obj_mask)
                    # cv2.imwrite(f"output/talha/image{frame_n},{obj_i}.jpg", obj_image)

                    _forg = np.float32(obj_image)

                    _alpha = np.float32(obj_mask) / 255

                    _back = background_float32[obj_bbox.ya():obj_bbox.yb(), obj_bbox.xa():obj_bbox.xb()]

                    _forg = cv2.multiply(_alpha, _forg)
                    _back = cv2.multiply(1.0 - _alpha, _back)

                    output = cv2.add(_forg, _back)
                    if output is not None:
                        out_frame[obj_bbox.ya():obj_bbox.yb(), obj_bbox.xa():obj_bbox.xb()] = output
                        # font = cv2.FONT_HERSHEY_SIMPLEX
                        # cv2.putText(out_frame, f"{obj.serial},{len(obj.images)}, {good_bbox_i}",
                        #             (obj_bbox.xa() + obj_bbox.w // 2, obj_bbox.ya() + obj_bbox.h // 2),
                        #             font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

        out_frame_path = os.path.join(out_frame_folder, f"frame{frame_n:06d}.jpg")
        cv2.imwrite(out_frame_path, out_frame)
        # cv2.imshow("image", out_frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # if cv2.wai    tKey(1) & 0xFF == ord('q'):
        #     break
    cv2.destroyAllWindows()


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


# def background_sub(in_frame_folder, out_frame_folder, start=1, end=None, show_video=False):
#     in_frame_folder = os.path.abspath(in_frame_folder)
#     out_frame_folder = os.path.abspath(out_frame_folder)
#
#     sorted_frames_filenames = sorted(os.listdir(in_frame_folder))
#
#     frame_filename = os.path.join(in_frame_folder, sorted_frames_filenames[0])
#     background = cv2.imread(frame_filename)
#     background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
#
#     for frame_n, frame_filename in enumerate(sorted_frames_filenames, start=start + 1):
#         frame_filename = os.path.join(in_frame_folder, frame_filename)
#         frame = cv2.imread(frame_filename)
#         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)
#
#         difference = cv2.absdiff(background_gray, frame_gray)
#         _, difference = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)
#
#         ret, thresh = cv2.threshold(difference, 127, 255, 0)
#         im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         cv2.drawContours(im2, contours, -1, (0, 255, 0), 3)
#
#         # cv2.imwrite(f"{out_frame_folder}/frame{frame_n:06d}.jpg", difference)
#         # print(frame_n)
#         if show_video:
#             cv2.imshow("image", im2)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         # else:
#         #     cv2.waitKey(0)
#         #     cv2.destroyAllWindows()
#         if frame_n == end:
#             break
#
#     cv2.destroyAllWindows()


def create_summary(in_video, allowed_objects=["car"]):
    in_video = os.path.abspath(in_video)
    video_hash = sha1(in_video)
    new_dir_path = os.path.join("processing", video_hash)

    # Create processing directory
    if not os.path.exists("processing"):
        print("1. Processing Directory Creation Initiated.")
        os.mkdir("processing")

    # Create video container directory
    if not os.path.exists(new_dir_path):
        print("2. Video Container Directory Creation Initiated.")
        os.mkdir(new_dir_path)

    # Create frames directory
    frames_dir_path = os.path.join(new_dir_path, "frames")
    if not os.path.exists(frames_dir_path):
        print("3. Video To Frames Conversion Initiated.")
        os.mkdir(frames_dir_path)
        os.system(f"ffmpeg -i {in_video} -vf fps=30 -qscale:v 5 {os.path.join(frames_dir_path, 'frame%06d.jpg')}")

    # Create Result directory
    result_dir_path = os.path.join(new_dir_path, "results")
    if not os.path.exists(result_dir_path):
        print("4. Result directory Creation Initiated.")
        os.mkdir(result_dir_path)

    # Compute Background Frame
    background_path = os.path.join(result_dir_path, "background.jpeg")
    if not os.path.exists(background_path):
        print("5. Background Extraction Initiated.")
        extract_background(frames_dir_path, background_path)

    # Detect Objects using YOLO v3
    detection_results_path = os.path.join(result_dir_path, "detections.tar.gz")
    if not os.path.exists(detection_results_path):
        print("5. Object Detection Initiated.")
        yolo_detector(frames_dir_path, detection_results_path)

    # Create foreground mask directory
    foreground_mask_dir_path = os.path.join(new_dir_path, "forground_masks")
    if not os.path.exists(foreground_mask_dir_path):
        print("6. Foreground Computation Initiated.")

        os.mkdir(foreground_mask_dir_path)

        # Create Foreground Masks
        background_sub(in_frame_folder=frames_dir_path,
                       out_frame_folder=foreground_mask_dir_path)

    # Track Object (Path)
    _store_path = os.path.join(result_dir_path, "object_store.pickled")
    _store_data_path = os.path.join(result_dir_path, "store")

    if not os.path.exists(_store_path) or not os.path.exists(_store_data_path):
        print("7. Object Tracking Initiated.")
        os.mkdir(_store_data_path)

        yolo_object_tracker(in_frame_folder=frames_dir_path,
                            in_detection_result_file=detection_results_path,
                            in_foreground_mask_folder=foreground_mask_dir_path,
                            allowed_objects=allowed_objects,
                            _store_path=_store_path,
                            _store_data_path=_store_data_path)

    # Create summarized video
    summarized_frames_path = os.path.join(new_dir_path, "summarized_frames")
    if not os.path.exists(summarized_frames_path):
        print("8. Summarized Video Creation Initiated.")

        os.mkdir(summarized_frames_path)

        process_yolo_background_sub(in_results_file=_store_path,
                                    in_background_file=background_path,
                                    out_frame_folder=summarized_frames_path,
                                    show_video=True,
                                    allowed_objects=allowed_objects)

    print("Summarized Video Created")


def main():
    create_summary("input/videos/ferozpurclip.mp4")
    # yolo_drawbbox("input/videos/frames", "output/ferozpur06012019_yolov3.tar.gz", "output/yolo", show_video=True)
    # background_ext(in_frame_folder="input/videos/frames",
    #                  out_frame_folder="output/ferozpur_bs_without_transformation")
    # yolo_deeplab(in_frame_folder="input/videos/frames", in_detection_result_file="output/ferozpur23122018(1).tar.gz",
    #              in_mask_file="input/deeplabmasks.tar.gz", out_frame_folder="output/frames03012019", show_video=False)
    #
    # yolo_background_sub(in_frame_folder="input/videos/frames",
    #                     in_detection_result_file="output/ferozpur06012019_yolov3.tar.gz",
    #                     in_foreground_mask_folder="output/ferozpur_bs",
    #                     out_frame_folder="output/frames03012019", show_video=False,
    #                     allowed_objects=["car"])
    # yolo_object_tracker(in_frame_folder="input/videos/frames",
    #                     in_detection_result_file="output/ferozpur06012019_yolov3.tar.gz",
    #                     in_foreground_mask_folder="output/ferozpur_bs",
    #                     allowed_objects=["car"])
    # process_yolo_background_sub(in_results_file="results.pickled",
    #                             out_frame_folder="output/frames03012019", show_video=True,
    #                             allowed_objects=["car"])

    # video = BATRVideoCapture("/home/meow/Desktop/slides/videos/ferozpur.mp4")
    # video.create_substantial_motion_video("output/ferozpur_substantial")


main()

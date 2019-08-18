
# def add_tracker_to_yolo(in_detection_result_file):
#     in_detection_result_file = os.path.abspath(in_detection_result_file)

#     pickler = BATRPickle(in_file=in_detection_result_file)
#     output_pickler = BATRPickle(out_file="detections_with_tracker.tar.gz")

#     for frame_n in range(1, 1802):
#         print("Frame #", frame_n)
#         detected_objects = pickler.unpickle(f"frame{frame_n:06d}")
#         detected_objects_fixed = []
#         for obj in detected_objects:
#             detected_objects_fixed.append(DetectedObject(obj.type, obj.probability,
#                                                          obj.x, obj.y, obj.w, obj.h))
#             obj.tracker = None
#         output_pickler.pickle(detected_objects_fixed, f"frame{frame_n:06d}")


# def fill_missing_detection(in_frame_folder, in_detection_result_file):
#     in_detection_result_file = os.path.abspath(in_detection_result_file)
#
#     pickler = BATRPickle(in_file=in_detection_result_file)
#     sorted_frames_abs_filenames = sorted([os.path.join(in_frame_folder, filename) for filename in
#                                           os.listdir(in_frame_folder)])
#
#     start = 1
#
#     # output_pickler = BATRPickle(out_file="detections.tar.gz")
#
#     last_frame_detected_obj = []
#     last_serial = 0
#     for frame_n, frame_filename in enumerate(sorted_frames_abs_filenames[start:], start=start):
#         print(f"Frame # {frame_n}\n{'-' * 10}")
#
#         frame = cv2.imread(frame_filename)
#
#         detected_objects = pickler.unpickle(f"frame{frame_n:06d}")
#
#         for obj in last_frame_detected_obj:
#             nn_index, nn_distance = nearest_neighbor(obj, detected_objects)
#             found_object = detected_objects[nn_index]
#
#             if nn_distance > 0.5 and not math.isinf(nn_distance) and hasattr(obj, "serial"):
#                 set_tracker(found_object, frame)
#                 found_object.serial = obj.serial
#                 # cv2.rectangle(frame, found_object.pt_a(), found_object.pt_b(), COLOR['green'], 4)
#             else:
#                 # Lost
#                 if obj.tracker is not None:
#                     ok, bbox = obj.tracker.update(frame)
#                     bbox = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
#                     cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]),
#                                   COLOR['yellow'], 6)
#
#                     if ok:
#                         lost_and_found = DetectedObject(obj.type, obj.probability,
#                                                         bbox[0], bbox[1], bbox[2], bbox[3])
#                         lost_and_found.serial = obj.serial
#                         lost_and_found.tracker = obj.tracker
#                         detected_objects.append(lost_and_found)
#
#                         cv2.rectangle(frame, lost_and_found.pt_a(), lost_and_found.pt_b(), COLOR['yellow'], 6)
#                 else:
#                     cv2.rectangle(frame, obj.pt_a(), obj.pt_b(), COLOR['white'], 6)
#
#                     # Detected First Time
#                     set_tracker(obj, frame)
#                     obj.serial = last_serial
#                     last_serial += 1
#
#         last_frame_detected_obj = detected_objects.copy()
#
#         # write_text(frame, f"Frame#{frame_n}", (100, 100))
#         detected_objects_without_trackers = detected_objects.copy()
#         for obj in detected_objects_without_trackers:
#             object_bbox_check(frame, obj)
#             write_text(frame, obj.serial, (obj.cx(), obj.cy()))
#
#             cv2.rectangle(frame, obj.pt_a(), obj.pt_b(), COLOR['green'], 4)
#         # output_pickler.pickle(detected_objects_without_trackers, f"frame{frame_n:06d}")
#
#         cv2.imshow("f", frame)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     # del pickler


# def yolo_object_tracker(in_frame_folder, in_detection_result_file, in_foreground_mask_folder, _store_path,
#                         _store_data_path, start=1, end=None):
#     in_frame_folder = os.path.abspath(in_frame_folder)
#     in_foreground_mask_folder = os.path.abspath(in_foreground_mask_folder)
#     in_detection_result_file = os.path.abspath(in_detection_result_file)
#
#     pickler = BATRPickle(in_file=in_detection_result_file)
#
#     sorted_frames_filenames = sorted(os.listdir(in_frame_folder))
#
#     store = []
#
#     for frame_n, frame_filename in enumerate(sorted_frames_filenames, start=start):
#         frame_file_abs_path = os.path.join(in_frame_folder, frame_filename)
#         foreground_mask_abs_path = os.path.join(in_foreground_mask_folder, frame_filename)
#         detected_objects = pickler.unpickle(f"frame{frame_n:06d}")
#         frame = cv2.imread(frame_file_abs_path)
#         mask = cv2.imread(foreground_mask_abs_path)
#
#         for obj in detected_objects:
#             # fixing a bug where obj.type have prefix of "b'" and postfix of "'"
#             # this bug comes from Yolo V3 detection layer
#             obj.type = str(obj.type)[2:-1]
#
#             obj_image = frame[obj.ya():obj.yb(), obj.xa():obj.xb()]
#             obj_mask = mask[obj.ya():obj.yb(), obj.xa():obj.xb()]
#
#             if obj_mask.size > 0 and obj_image.size > 0:
#                 track_object(obj=obj, obj_mask=obj_mask, obj_image=obj_image, _store=store,
#                              _frame_n=frame_n, _store_data_path=_store_data_path)
#
#         print(f"Frame# {frame_n}")
#
#         if frame_n == end:
#             break
#     with open(_store_path, "wb") as f:
#         pickle.dump(store, f)
#
#     cv2.destroyAllWindows()


# def notify_progress_observer(progress_observer, obserable):
#     while True:
#         progress = progress_observer.notify(obserable)
#         print(progress)
#         if progress == 100:
#             print("Returning")
#             return 0
#         time.sleep(1)
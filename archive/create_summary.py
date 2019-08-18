# def create_summary(in_video, allowed_objects=None,
#                    force_video_to_frames=False, force_background_extraction=False,
#                    force_detection=False, force_foreground_computation=False,
#                    force_object_tracking=False, force_create_summarized_video=False):
#     if allowed_objects is None:
#         allowed_objects = ["car"]
#
#     in_video = os.path.abspath(in_video)
#     video_hash = sha1(in_video)
#     new_dir_path = os.path.join("processing", video_hash)
#
#     # Create processing directory
#     if not os.path.exists("processing"):
#         print("1. Processing Directory Creation Initiated.")
#         os.mkdir("processing")
#
#     # Create video container directory
#     if not os.path.exists(new_dir_path):
#         print("2. Video Container Directory Creation Initiated.")
#         os.mkdir(new_dir_path)
#
#     # Create frames directory
#     frames_dir_path = os.path.join(new_dir_path, "frames")
#     if not os.path.exists(frames_dir_path) or force_video_to_frames:
#         print("3. Video To Frames Conversion Initiated.")
#         os.mkdir(frames_dir_path)
#         options = "-start_number 0 -b:v 10000k -vsync 0 -an -y -q:v 5"
#         os.system(f"ffmpeg -i {in_video} {options} {os.path.join(frames_dir_path, 'frame%06d.jpg')}")
#
#     # Create Result directory
#     result_dir_path = os.path.join(new_dir_path, "results")
#     if not os.path.exists(result_dir_path):
#         print("4. Result directory Creation Initiated.")
#         os.mkdir(result_dir_path)
#
#     # Background Extraction
#     background_path = os.path.join(result_dir_path, "background.jpeg")
#     if not os.path.exists(background_path) or force_background_extraction:
#         print("5. Background Extraction Initiated.")
#         extract_background(frames_dir_path, background_path)
#
#     # Detect Objects
#     detection_results_path = os.path.join(result_dir_path, "detections.tar.gz")
#     if not os.path.exists(detection_results_path) or force_detection:
#         print("5. Object Detection Initiated.")
#         detect_and_save(frames_dir_path, detection_results_path)
#
#     # Create foreground mask directory
#     foreground_mask_dir_path = os.path.join(new_dir_path, "forground_masks")
#     if not os.path.exists(foreground_mask_dir_path) or force_foreground_computation:
#         print("6. Foreground Computation Initiated.")
#
#         if not os.path.exists(foreground_mask_dir_path):
#             os.mkdir(foreground_mask_dir_path)
#
#         # Create Foreground Masks
#         background_sub(in_frame_folder=frames_dir_path,
#                        out_frame_folder=foreground_mask_dir_path)
#
#     # Track Object
#     _store_path = os.path.join(result_dir_path, "object_store.pickled")
#     _store_data_path = os.path.join(result_dir_path, "store")
#
#     if not os.path.exists(_store_path) or not os.path.exists(_store_data_path) or force_object_tracking:
#         print("7. Object Tracking Initiated.")
#         if not os.path.exists(_store_data_path):
#             os.mkdir(_store_data_path)
#
#         # yolo_object_tracker(in_frame_folder=frames_dir_path,
#         #                     in_detection_result_file=detection_results_path,
#         #                     in_foreground_mask_folder=foreground_mask_dir_path,
#         #                     _store_path=_store_path,
#         #                     _store_data_path=_store_data_path)
#
#     # Create summarized video
#     summarized_frames_path = os.path.join(new_dir_path, "summarized_frames")
#     if not os.path.exists(summarized_frames_path) or force_create_summarized_video:
#         print("8. Summarized Video Creation Initiated.")
#
#         if not os.path.exists(summarized_frames_path):
#             os.mkdir(summarized_frames_path)
#
#         process_yolo_background_sub(in_results_file=_store_path,
#                                     in_background_file=background_path,
#                                     out_frame_folder=summarized_frames_path,
#                                     allowed_objects=allowed_objects)
#
#     improved_fill_missing_detection(in_frame_folder=frames_dir_path, in_detection_result_file=detection_results_path)
#     print("Summarized Video Created")
def fill_missing_detection_using_features(in_frame_folder, in_detection_result_file):
    in_detection_result_file = os.path.abspath(in_detection_result_file)

    pickler = BATRPickle(in_file=in_detection_result_file)
    sorted_frames_filenames = sorted(os.listdir(in_frame_folder))

    last_frame_detected_obj = []

    start = 154
    orb = cv2.ORB_create(nfeatures=10000, scoreType=cv2.ORB_FAST_SCORE)

    # algorithm = FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    for frame_n, frame_filename in enumerate(sorted_frames_filenames[start:], start=start):
        print(f"Frame # {frame_n}")
        print("-" * 10)
        frame_file_abs_path = os.path.join(in_frame_folder, frame_filename)

        frame = cv2.imread(frame_file_abs_path)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detected_objects = pickler.unpickle(f"frame{frame_n:06d}")

        for obj in detected_objects:
            obj_img = frame_gray[obj.ya():obj.yb(), obj.xa():obj.xb()]
            kp, des = orb.detectAndCompute(obj_img, None)
            obj.keypoints = kp.copy()
            obj.descriptors = np.float32(des)
            obj.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            # if len(obj.keypoints) < 2:
            #     cv2.imshow("f", obj_img)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()

        detected_objects = [obj for obj in detected_objects if len(obj.keypoints) > 0]

        for obj in last_frame_detected_obj:
            # fixing a bug where obj.type have prefix of "b'" and postfix of "'"
            # this bug comes from Yolo V3 detection layer
            if not hasattr(obj, "interpolated"):
                obj.type = str(obj.type)[2:-1]

            if len(obj.keypoints) < 2:
                obj_img = frame_gray[obj.ya():obj.yb(), obj.xa():obj.xb()]
                cv2.imshow("im", obj_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                cv2.rectangle(frame, (obj.xa(), obj.ya()), (obj.xb(), obj.yb()), (255, 255, 255), 4)
                continue

            closest_neighbour = None

            for _obj in detected_objects:
                if len(_obj.descriptors) < 2:
                    continue

                matches = flann.knnMatch(obj.descriptors, _obj.descriptors, k=2)
                good = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good.append(m)
                if closest_neighbour is None:
                    closest_neighbour = _obj, len(good)
                else:
                    if closest_neighbour[1] < len(good):
                        closest_neighbour = _obj, len(good)
            print("Best Match", obj, closest_neighbour)
            if closest_neighbour is not None:
                cv2.rectangle(frame, (obj.xa(), obj.ya()), (obj.xb(), obj.yb()), (0, 255, 0), 4)
            else:
                cv2.rectangle(frame, (obj.xa(), obj.ya()), (obj.xb(), obj.yb()), COLOR['red'], 4)
                cv2.putText(frame, obj.type, (obj.xa() + int(obj.w / 2), obj.ya() + int(obj.h / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR['red'], 2)

        cv2.imshow("f", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        last_frame_detected_obj = detected_objects.copy()


def fill_missing_detection_using_features_and_optflow(in_frame_folder, in_foreground_mask_folder, in_detection_result_file,
                           in_optflow_folder):
    in_detection_result_file = os.path.abspath(in_detection_result_file)

    pickler = BATRPickle(in_file=in_detection_result_file)
    sorted_frames_filenames = sorted(os.listdir(in_frame_folder))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    last_frame_detected_obj = []

    start = 154
    orb = cv2.ORB()

    for frame_n, frame_filename in enumerate(sorted_frames_filenames[start:], start=start):
        # print(frame_filename)
        print(f"Frame # {frame_n}")
        print("-" * 10)
        frame_file_abs_path = os.path.join(in_frame_folder, frame_filename)
        foreground_mask_abs_path = os.path.join(in_foreground_mask_folder, frame_filename)

        frame = cv2.imread(frame_file_abs_path)
        orb.detectAndCompute()
        # mask = cv2.imread(foreground_mask_abs_path)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

        detected_objects = pickler.unpickle(f"frame{frame_n:06d}")

        for obj in last_frame_detected_obj:
            # fixing a bug where obj.type have prefix of "b'" and postfix of "'"
            # this bug comes from Yolo V3 detection layer
            if not hasattr(obj, "interpolated"):
                obj.type = str(obj.type)[2:-1]

            if obj.type in ["motorbike", "person"]:
                cv2.rectangle(frame, (obj.xa(), obj.ya()), (obj.xb(), obj.yb()), (0, 0, 255), 4)
                cv2.putText(frame, obj.type,  (obj.xa() + int(obj.w/2), obj.ya() + int(obj.h/2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                continue

            # print(f"{obj.bbox()}'s nearest neighbour is  {nearest_neighbor_dist(obj, detected_objects)} apart")
            nearest_neighbor_distance = nearest_neighbor_dist(obj, detected_objects)
            if nearest_neighbor_distance > 0.5:
                # print(":) FOUND!")
                if not hasattr(obj, "interpolated"):
                    cv2.rectangle(frame, (obj.xa(), obj.ya()), (obj.xb(), obj.yb()), (0, 255, 0), 4)
                else:
                    cv2.rectangle(frame, (obj.xa(), obj.ya()), (obj.xb(), obj.yb()), (0, 255, 255), 4)

            else:
                # Orphaned Object
                adopted = obj
                opt_x = np.average(flow[adopted.xa():adopted.xb(), adopted.ya():adopted.yb(), 0])
                opt_y = np.average(flow[adopted.xa():adopted.xb(), adopted.ya():adopted.yb(), 1])

                if math.isnan(opt_x) or math.isnan(opt_y):
                    opt_x = 0
                    opt_y = 0
                else:
                    opt_x = round(opt_x, 1)
                    opt_y = round(opt_y, 1)

                print("OBJ", adopted.x, adopted.y)
                print("OPT", opt_x, opt_y)

                adopted.x = int(adopted.x + round(opt_x, 2))
                adopted.y = int(adopted.y + round(opt_y, 2))

                adopted.interpolated = True
                detected_objects.append(adopted)
                cv2.rectangle(frame, (obj.xa(), obj.ya()), (obj.xb(), obj.yb()), (0, 255, 255), 4)
        last_frame_detected_obj = detected_objects.copy()
        # if len(orphaned_obj):
        #     print("Orphan ed Objects", orphaned_obj)
        cv2.imshow("f", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

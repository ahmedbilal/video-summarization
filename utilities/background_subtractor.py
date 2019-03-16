import os

import cv2


def background_sub(in_frame_folder, out_frame_folder, thresholded=True, start=0, end=None):
    """
    Output background subtractor output to path specified by "out_frame_folder" or 2nd argument
    :param in_frame_folder: Path of Folder Containing Input Frames
    :param out_frame_folder: Path of Folder where output of background subtractor is saved
    :param thresholded: is output be thresholded? (default=True)
    :param start: Starting Frame Number (default: 0)
    :param end: Ending Frame Number (default: last frame)
    """
    in_frame_folder = os.path.abspath(in_frame_folder)
    out_frame_folder = os.path.abspath(out_frame_folder)

    sorted_frames_filenames = sorted(os.listdir(in_frame_folder))

    if thresholded:
        foreground_map = cv2.createBackgroundSubtractorKNN(dist2Threshold=500)
    else:
        foreground_map = cv2.createBackgroundSubtractorKNN()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    for frame_n, frame_filename in enumerate(sorted_frames_filenames, start=start):
        frame_filename = os.path.join(in_frame_folder, frame_filename)
        frame = cv2.imread(frame_filename)

        mask = foreground_map.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

        cv2.imwrite(f"{out_frame_folder}/frame{frame_n:06d}.jpg", mask)

        if frame_n == end:
            break

import cv2
import os
import glob
import tempfile
import tarfile
from video_summarization.pyflow.optical_flow import optical_flow
import multiprocessing as mp

from vi3o import Video

CONFIG = dict()
CONFIG["offset"] = 0
CONFIG["end"] = 50

CONFIG["input_video_file"] = "input/videos/wales_shortened.mp4"


CONFIG["display_optical_flow"] = False

CONFIG["create_compressed_file"] = True


def compute(offset, end):
    CONFIG["output_compressed_file"] = "optical_flow{}-{}.tar.gz".format(offset, end)

    video = Video(CONFIG["input_video_file"])
    compressed_file = None
    tmp_dir = None

    if CONFIG["create_compressed_file"]:
        tmp_dir = tempfile.TemporaryDirectory()
        compressed_file = tarfile.open(CONFIG["output_compressed_file"], "w:gz")

    for current_frame_n in range(offset, end):
        try:
            crnt = video[current_frame_n]
            nxt = video[current_frame_n + 1]
        except StopIteration:
            break
        print("Meow")
        opt = optical_flow(nxt, crnt)
        print("Stupid Optical Flow")
        if CONFIG["display_optical_flow"]:
            cv2.imshow("Optical Flow", opt)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if CONFIG["create_compressed_file"]:
            cv2.imwrite("{}/optical_flow{}-{}.jpg".format(tmp_dir.name, current_frame_n, current_frame_n + 1), opt)

        print("Counter #", current_frame_n)

    os.chdir(tmp_dir.name)

    if CONFIG["create_compressed_file"]:
        for _files in glob.glob("*".format(tmp_dir.name)):
            compressed_file.add(_files, recursive=False)
        compressed_file.close()
        tmp_dir.cleanup()


def main():
    p = mp.Pool(6)
    print(p.starmap(compute, [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]))


main()

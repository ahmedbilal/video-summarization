import cv2

OUTPUT_FOLDER = "frames"
VIDEO = "flowers.mp4"

video = cv2.VideoCapture(VIDEO) # create a VideoCapture object and use "flowers.mp4" as input
success, image = video.read()  # return read_status and next_frame_image
counter = 0
while success:
    cv2.imwrite("{OUTPUT_FOLDER}/frame_{count}.jpg".format(OUTPUT_FOLDER = OUTPUT_FOLDER,
                                                           count = counter), image)
    success, image = video.read()  # return read_status and next_frame_image
    counter += 1

import cv2
import numpy as np


from utilities.utils import BATRVideoCapture, BATRPickle, BATRTar

offset = 0
counter = 0
video = BATRVideoCapture("input/videos/wales_shortened.mp4", offset)
optical_flow_tar = BATRTar.open("input/optical_flow0-750.tar.xz", "r:xz")
output_video = cv2.VideoWriter("output/optical_flow_gradient.mp4",
                               cv2.VideoWriter_fourcc(*"mp4v"), 30, video.dimensions)
run_video = True


for file in optical_flow_tar.allextractedfiles():
    content = np.fromstring(file.read(), np.uint8)
    image = np.asarray(cv2.imdecode(content, cv2.IMREAD_COLOR))
    gradient = np.gradient(image)
    hypot = np.uint8(np.hypot(gradient[0], gradient[1]))
    output_video.write(hypot)

    cv2.imshow("image", hypot)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    counter += 1

    if counter == 250:
        break

cv2.destroyAllWindows()
output_video.release()

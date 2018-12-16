import cv2
import numpy as np
import tarfile

from utilities.utils import BATRPickle

optical_flow_tar = tarfile.open("input/optical_flow0-750.tar.xz", "r:xz")
pickled_masks = BATRPickle("input/maskrcnn0-5000.tar.gz")
run_video = True


prev_frame = optical_flow_tar
for i in range(1, 5000):

for member in :
    file = optical_flow_tar.extractfile(member)
    content = np.fromstring(file.read(), np.uint8)
    image = np.asarray(cv2.imdecode(content, cv2.IMREAD_COLOR))
    cv2.imshow("optical flow", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()


# for file in optical_flow_tar.allextractedfiles():
#     content = np.fromstring(file.read(), np.uint8)
#     image = np.asarray(cv2.imdecode(content, cv2.IMREAD_COLOR))
#     gradient = np.gradient(image)
#     hypot = np.uint8(np.hypot(gradient[0], gradient[1]))
#     output_video.write(hypot)
#
#     cv2.imshow("image", hypot)
#     if cv2.waitKey(30) & 0xFF == ord('q'):
#         break
#
# cv2.destroyAllWindows()
# output_video.release()

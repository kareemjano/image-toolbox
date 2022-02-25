import cv2
from facenet_pytorch import MTCNN
import numpy as np
from PIL import Image
import torch

def angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    tan = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan))


def get_rotation_matrix(p1, p2):
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M

def image_deep_alignment(img, transform_kind="crop", precomputed_detection=None, precomputed_landmarks=None,
                         compute_landmarks=True):
    # convert image to np array
    img = np.array(img)

    # compute bounding box and landmarks
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # initialize detector
    face_detector = MTCNN(device=device)
    face_detector.select_largest = True
    detections, probs, landmarks = None, None, None
    # detect landmark points
    if not compute_landmarks:
        detections, probs = face_detector.detect(img, landmarks=False)
    else:
        detections, probs, landmarks = face_detector.detect(img, landmarks=True)

    transformed = img

    if detections is not None:

        x, y, x2, y2 = int(detections[0][0]), int(detections[0][1]), int(detections[0][2]), int(detections[0][3])
        h = img.shape[0]
        w = img.shape[1]

        # rotation transformation
        left_eye = landmarks[0][0]
        right_eye = landmarks[0][1]
        rotation = get_rotation_matrix(left_eye, right_eye)
        transformed = cv2.warpAffine(img, rotation, img.shape[:2], flags=cv2.INTER_CUBIC)

        # crop the bounding boxes and expand the box by a factor of 1/3
        # elif transform_kind == "crop":
        #     y = max(0, y - int((y2 - y) * 1 / 3))
        #     y2 = min(y2 + int((y2 - y) * 1 / 3), h - 1)
        #     x = max(x - int((x2 - x) * 1 / 3), 0)
        #     x2 = max(x2 + int((x2 - x) * 1 / 3), w - 1)
        #
        #     return Image.fromarray(img[y:y2, x:x2, :]), detections, landmarks

    return Image.fromarray(transformed), detections, landmarks
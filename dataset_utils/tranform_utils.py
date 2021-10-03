import numpy as np
from ..image_utils.face_utils import image_deep_alignment
from PIL import Image

class FaceAlignTransform(object):
    """
    Align the face by crop only (SIMPLE kind) or crop and rotation (ROTATION kind)
    """

    def __init__(self, shape):
        self.shape = shape

    def __call__(self, img):
        return self.crop_and_resize(img)

    def crop_and_resize(self, img):
        img, detections, landmarks = image_deep_alignment(img)

        # rotate the image
        img, _, _ = image_deep_alignment(img, precomputed_detection=detections,
                                         precomputed_landmarks=landmarks)
        return img

class ToNumpy(object):
    """
    Align the face
    """

    def __init__(self):
        pass

    def __call__(self, img):
        return np.array(img)
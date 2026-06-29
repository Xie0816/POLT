"""Image loading helpers for POLT camera frames."""

import cv2
from PIL import Image


def load_pil_image(path):
    return Image.open(path)


def load_bgr_image(path):
    return cv2.imread(path)


def load_rgb_array(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

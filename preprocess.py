import cv2


def get_CLAHE(clipLimit=10.0, tileGridSize=(16, 16)):
    return cv2.createCLAHE(clipLimit, tileGridSize).apply


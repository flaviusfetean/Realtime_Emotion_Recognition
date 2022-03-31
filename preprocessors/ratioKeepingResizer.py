import cv2
import imutils

class rkResizer:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):

        (h, w) = image.shape[:2]
        dh = 0
        dw = 0

        if w < h:
            image = imutils.resize(image, width=self.width, inter=self.inter)
            (h, w) = image.shape[:2]
            dh = int((h - w)/2)
        else:
            image = imutils.resize(image, height=self.height, inter=self.inter)
            (h, w) = image.shape[:2]
            dw = int((w - h)/2)

        image = image[dh:h - dh, dw:w - dw]

        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
import numpy as np
import cv2

class CropPreprocessor:
    def __init__(self, width, height, horiz=True, inter=cv2.INTER_AREA):

        self.width = width
        self.height = height
        self.horiz = horiz
        self.inter = inter

    def preprocess(self, image):

        crops = []

        (h, w) = image.shape[:2]

        crop_coords = [
            [0, 0, self.width, self.height],
            [w - self.width, 0, w, self.height],
            [0, h - self.height, self.width, h],
            [w - self.width, h - self.height, w, h]
        ]

        midW = int(0.5 * (w - self.width))
        midH = int(0.5 * (h - self.height))
        crop_coords.append([midW, midH, w - midW, h - midH])

        for (startX, startY, endX, endY) in crop_coords:

            crop = image[startY:endY, startX:endX]
            crop = cv2.resize(crop, (self.width, self.height),
                              interpolation=self.inter)
            crops.append(crop)

        if self.horiz:
            mirrored = [cv2.flip(img, 1) for img in crops]
            crops.extend(mirrored) #.extend method "concatenates" 2 array, as opposed
            # to ".append" that adds an element at the end of an array
            #we can t use .append here, as we would loop through an array, if we'd append,
            #we would loop infinitely, so we loop once and save thngs in an array, then concatenate

        return np.array(crops)
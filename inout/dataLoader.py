import numpy as np
import cv2
import os


class datasetLoader:
    def __init__(self, preprocessors=None):
        if preprocessors == None:
            self.preprocessors = []
        else:
            self.preprocessors = preprocessors

    def load_dataset(self, imagePaths, verbose=-1):
        data = []
        labels = []

        if self.preprocessors is not None:
            for (i, img_path) in enumerate(imagePaths):
                image = cv2.imread(img_path)
                label = img_path.split(os.path.sep)[-2]

                for p in self.preprocessors:
                    image = p.preprocess(image)

                data.append(image)
                labels.append(label)
                if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                    print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))

        return (np.array(data), np.array(labels))

    def load_flowers(self, imagePaths, verbose = -1):
        data = []
        labels = []

        j = 0;

        if self.preprocessors is not None:
            for (i, img_path) in enumerate(imagePaths):
                image = cv2.imread(img_path)
                label = str(j)

                for p in self.preprocessors:
                    image = p.preprocess(image)

                data.append(image)
                labels.append(label)
                if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                    print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))
                if (i + 1) % 80 == 0:
                    j += 1

        return (np.array(data), np.array(labels))
from tensorflow.keras.utils import to_categorical
import numpy as np
import h5py

class HDF5DatasetGenerator:
    def __init__(self, dbPath, batchSize, preprocessors=None,
                 binarize=True, num_classes=2):
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.binarize = binarize
        self.num_classes = num_classes

        self.db = h5py.File(dbPath)
        self.numImages = self.db["labels"].shape[0]

    def generator(self, passes=np.inf):  #we create a generator class to pass to the
        #fit method of a keras model in ordr to constantly yield
        #batches of data read from a h5py database
        epochs = 0

        while epochs < passes:
            for i in np.arange(0, self.numImages, self.batchSize):

                #Remmber that these images are taken from the actual external physical
                #memory, only they are easily accessed in an array-like manner
                #using the h5py library
                images = self.db["images"][i: i + self.batchSize]
                labels = self.db["labels"][i: i + self.batchSize]

                if self.binarize:
                    labels = to_categorical(labels, self.num_classes)

                if self.preprocessors is not None:

                    proc_images = []

                    for image in images:
                        for p in self.preprocessors:
                            image = p.preprocess(image)

                        proc_images.append(image)

                    images = np.array(proc_images)

                yield (images, labels)

            epochs += 1

    def close(self):
        self.db.close()

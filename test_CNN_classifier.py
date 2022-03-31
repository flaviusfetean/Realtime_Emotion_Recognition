import config
from preprocessors.imagetoarraypreprocessor import ImageToArrayPreprocess
from inout.hdf5datasetgenerator import HDF5DatasetGenerator
from tensorflow.keras.models import load_model

iap = ImageToArrayPreprocess()

testGen = HDF5DatasetGenerator(config.TEST_HDF5, 32, preprocessors=[iap], num_classes=6)

model = load_model("outputs/model")

(loss, acc) = model.evaluate(testGen.generator(), steps=testGen.numImages // 32,
                             max_queue_size=64)

print("[INFO] accuracy: {}".format(acc * 100))

testGen.close()
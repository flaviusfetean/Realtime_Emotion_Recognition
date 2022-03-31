import matplotlib
matplotlib.use("Agg")

from neuralnet.SimpleConvoNet import Categorical_ConvoNet as cnn
from preprocessors.imagetoarraypreprocessor import ImageToArrayPreprocess
from helpers.callbacks.trainingmonitor import TrainingMonitor
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
from inout.hdf5datasetgenerator import HDF5DatasetGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
import config
import tensorflow.keras.backend as K
import os

def build_mini_VGG():

    layers = ["Conv2D", "Batch Normalization", "Conv2D", "Batch Normalization", "Pooling",
              "Conv2D", "Batch Normalization", "Conv2D", "Batch Normalization", "Pooling",
              "Conv2D", "Batch Normalization", "Conv2D", "Batch Normalization", "Pooling",
              "Dense", "Dropout", "Dense", "Dropout", "Dense"]
    sizes = [[32, (3, 3)], 0, [32, (3, 3)], 0, ((2, 2), (2, 2)),
             [64, (3, 3)], 0, [64, (3, 3)], 0, ((2, 2), (2, 2)),
             [128, (3, 3)], 0, [128, (3, 3)], 0, ((2, 2), (2, 2)),
             64, 0.5, 64, 0.5, 6]
    activations = ["elu", "none", "elu", "none", "none",
                   "elu", "none", "elu", "none", "none",
                   "elu", "none", "elu", "none", "none",
                   "elu", "none", "elu", "none", "Softmax"]

    model = cnn.build(48, 48, 1, layers, sizes, activations, alpha_reg_intensity=0.0001,
                      augment_data=True, initializer="he_normal")

    return model

def lr_decay(epoch):
    learning_rate = 0.01
    if epoch >= 30:
        learning_rate /= 10
    if epoch >= 57:
        learning_rate /= 10

    return learning_rate

iap = ImageToArrayPreprocess()

trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 32, [iap], num_classes=config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 32, [iap], num_classes=config.NUM_CLASSES)

answer = input("[ACTION] Continue training? y/n")
start_epoch = 0

if answer == "n":
    print("[INFO] compiling model...")
    model = build_mini_VGG()
    opt = Adam() #SGD(learning_rate=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
else:
    print("[INFO] loading model...")
    model = load_model("outputs/model")

    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.learning_rate)))
    new_learning_rate = input("[ACTION] Please introduce a new larning rate: ")
    K.set_value(model.optimizer.learning_rate, float(new_learning_rate))
    start_epoch = input("[ACTION] Please introduce the starting epoch")

figPath = "outputs/training_data/graph_{}.png".format(os.getpid())
jsonPath = "outputs/training_data/raw_metrics.json"

callbacks = [
    ModelCheckpoint("outputs/model", monitor='val_loss', save_best_only=True),
    TrainingMonitor(figPath, jsonPath, startAt=int(start_epoch))]
    #LearningRateScheduler(lr_decay)

model.fit(trainGen.generator(), steps_per_epoch=trainGen.numImages // 32,
          validation_data=valGen.generator(), validation_steps=valGen.numImages // 32,
          epochs=75, initial_epoch=int(start_epoch), max_queue_size=64, verbose=1)

trainGen.close()
valGen.close()

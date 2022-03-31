from tensorflow.keras.callbacks import BaseLogger
import os
import matplotlib.pyplot as plt
import json
import numpy as np

class TrainingMonitor(BaseLogger):
    def __init__(self, figPath, jsonPath=None, startAt=0):
        super(TrainingMonitor).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt

    def on_train_begin(self, logs={}): #this function is automatically called at the beginning of training
        self.H = {} #history

        #if a json path was given
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath): #and it already exists
                self.H = json.loads(open(self.jsonPath).read())
                    #we initialize H with the values that were already there

            if self.startAt > 0: #if a starting epoch was given
                for k in self.H.keys(): #we loop through the history and just trim the existing data
                    self.H[k] = self.H[k][:self.startAt] # that would've come after it

    def on_epoch_end(self, epoch, logs={}): #this is automatically called at the end of each epoch
        for (k, v) in logs.items(): # basically we iterate through the data given by the keras' log
            l = self.H.get(k, []) #we create or search for an entry in the map H
            l.append(v) #k is a metric name, like "val_loss", and l is the array associated
            self.H[k] = l #with it in the map, v is the new value given by the log and we add it
        if self.jsonPath is not None: #if a file path is given, we also write the data in the given file
            f = open(self.jsonPath, "w")
            f.write(json.dumps(self.H))
            f.close()
        if len(self.H["loss"]) > 1: #if we passed the first epoch we can draw a graph
            N = np.arange(0, len(self.H["loss"])) #so we take the ticks
            plt.style.use("ggplot")

            fig, ax1 = plt.subplots()

            ax1.set_xlabel("Epoch #")
            ax1.set_ylabel("Accuracy")
            ax1.plot(N, self.H["val_accuracy"], label="val_acc", color="blue")
            ax1.plot(N, self.H["accuracy"], label="train_acc", color="black")

            ax2 = ax1.twinx()

            ax2.set_ylabel("Loss")
            ax2.plot(N, self.H["loss"], label="train_loss", color="red")  #and plot the metrics
            ax2.plot(N, self.H["val_loss"], label="val_loss", color="orange")

            plt.title("Training Loss and Accuracy [Epoch {}]".format(len(self.H["loss"])))
            plt.legend(loc=2)
            plt.savefig(self.figPath)  #these will be saved in the given file path
            plt.close()

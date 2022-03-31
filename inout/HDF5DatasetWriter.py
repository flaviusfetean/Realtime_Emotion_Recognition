import h5py
import os

class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey="images", bufSize=1000):

        if os.path.exists(outputPath):
            raise ValueError("The supplied output path already exists"
                             "and cannot be overwritten. Manually delete"
                             "the file and try again.", outputPath)

        self.db = h5py.File(outputPath, "w") #this will be the file where we dump our large info like a database
        self.data = self.db.create_dataset(dataKey, dims, dtype="float") #we create an instance of a dataset in this file for the images
        self.labels = self.db.create_dataset("labels", (dims[0],), dtype="int") #and one for the labels

        self.bufSize = bufSize #we will use this variable so that we never process more info than we want
        #the following declaration instantiates a map where we will store these temporary info
        #that we currently process in a way that is easy to access further
        self.buffer = {"data": [], "labels": []}
        #the h5py database is treated like an array so it is accesed using indexes. This var holds
        #the current index where the database points
        self.index = 0

    def add(self, rows, labels):

        # when we want to write into the database, we first put info into the buffer
        self.buffer["data"].extend(rows) #the data
        self.buffer["labels"].extend(labels) #along with the data's labels

        if len(self.buffer["data"]) >= self.bufSize: #but if the info contained in the buffer is too large
            #we will have to flush it (dump it into the physical external file and then
            #delete what was previously in the buffer)
            self.flush()

    def flush(self):

        #we have the current index of the database and we get in "i" the future
        #index after we dump our info
        i = self.index + len(self.buffer["data"])
        self.data[self.index:i] = self.buffer["data"] #access the database like an array
        self.labels[self.index:i] = self.buffer["labels"]

        self.index = i #update the index
        self.buffer = {"data": [], "labels": []} #empty the buffer

    def storeClassLabels(self, classLabels):

        #this is a one-time database creation if we want to

        dt = h5py.string_dtype(encoding='utf-8') #used because the type of elements
        #that will be stored in the database is not int or float or anything known,
        #instead, it's variable-length string type (not fixed, predictible, but variable)
        #so we need to specify it to the h5py object
        labelSet = self.db.create_dataset("label_names", (len(classLabels),), dtype=dt)
        labelSet[:] = classLabels

    def close(self):

        if len(self.buffer["data"]) > 0:
            self.flush()

        self.db.close()

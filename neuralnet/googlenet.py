from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.regularizers import L2

#all these could be just used as "from tf.keras import layers as L", and then
# call "L.Dropout" for example but it's far more visual like this

from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
import tensorflow.keras.layers.experimental.preprocessing as augment
import tensorflow as tf

class GoogLeNet_:
    @staticmethod
    def conv_module(inputModel, num_filters, x_kernel, y_kernel, stride,
                    chanDim, padding="same", reg=0.0005, name=None):
        """

        :param inputModel: the part of the network to which the conv_module is appended
        :param num_filters: the number of filters of the convolutional layer
        :param x_kernel: width of kernel
        :param y_kernel: height of kernel
        :param stride: the step taken by the kernel when swiping through
        :param chanDim: the placement of the depth channel (1:depth first, -1:depth last)
        :param padding: padding applied to the convolutional layer
        :param reg: L2 regularization intensity applied on weights
        :return: returns a conv_module with the input model as input
        """

        (convName, bnName, actName) = (None, None, None)

        if name is not None:
            convName = name + "_conv"
            bnName = name + "bn"
            actName = name + "act"

        x = Conv2D(num_filters, (x_kernel, y_kernel), strides=stride, padding=padding,
                   kernel_regularizer=L2(reg), name=convName)(inputModel)
        x = BatchNormalization(axis=chanDim, name=bnName)(x)
        x = Activation("relu", name=actName)(x)
        """As I said, tensorflow actually builds computational graphs, and the sequence of
        statements above takes the actual output of the "inputModel" (which is on its own
        a computational graph, and outputs something - i.e. the last layer) and inputs
        it to the conv layer we are adding, this being the shorthand notation for chaining
        computations in a computational graph (any subgraph of a computational graph can 
        be considered as a computation - so a single node from this point of view):                          """
            #outputOfWholeGraph = newLayer(parameters)(inputLayerOrGraph)

        return x

    @staticmethod
    def inception_module(inputModel, num1x1, num3x3Reduce, num3x3,
                         num5x5Reduce, num5x5, num1x1Proj, chanDim,
                         stage="", reg=0.0005):
        """

        :param inputModel: - same as in conv_module above -
        :param num1x1_filters: the number of filters with kernel dimension (1, 1)
        :param num3x3_filters: the number of filters with kernel dimension (3,3)
        :param chanDim: - same as in conv_module above -
        :return: returns an inception module appended to the inputModel
        """

        conv_1x1 = GoogLeNet_.conv_module(inputModel, num1x1, 1, 1, (1, 1),
                                          chanDim, reg=reg, name=stage + "_first")

        conv_3x3 = GoogLeNet_.conv_module(inputModel, num3x3Reduce, 1, 1, (1, 1),
                                          chanDim, reg=reg, name=stage + "_second1")
        conv_3x3 = GoogLeNet_.conv_module(conv_3x3, num3x3, 3, 3, (1, 1),
                                          chanDim, reg=reg, name=stage + "_second2")

        conv_5x5 = GoogLeNet_.conv_module(inputModel, num5x5Reduce, 1, 1, (1, 1),
                                          chanDim, reg=reg, name=stage + "third1")
        conv_5x5 = GoogLeNet_.conv_module(conv_5x5, num5x5, 5, 5, (1, 1),
                                          chanDim, reg=reg, name=stage + "third2")

        pool = MaxPooling2D((3, 3), strides=(1, 1), padding="same",
                            name=stage + "_pool")((inputModel))
        pool = GoogLeNet_.conv_module(pool, num1x1Proj, 1, 1, (1, 1),
                                      chanDim, reg=reg, name=stage + "fourth")

        x = concatenate([conv_1x1, conv_3x3, conv_5x5, pool], axis=chanDim, name=stage + "_mixed")

        return x

    @staticmethod
    def downsample_module(inputModule, num_filters, chanDim):
        """

        :param inputModule: - same as in conv_module above -
        :param num_filters: number of filters of the convolutional layer,
                            that will have (3, 3) kernel size with (2, 2) stride
        :param chanDim: - same as in conv_module above -
        :return: returns a downsample module appended to the inputModule,
                practically halving the output's size
        """

        conv_3x3 = GoogLeNet_.conv_module(inputModule, num_filters, 3, 3,
                                          (2, 2), chanDim, padding="valid")
        pool = MaxPooling2D((3, 3), strides=(2, 2))(inputModule)
        x = concatenate([conv_3x3, pool], axis=chanDim)

        return x

    @staticmethod
    def build_mini(width, height, depth, num_classes, aug=False):

        #initialize the depth dimension (channels - R, G, B = depth) to be last in ordering
        inputShape = (height, width, depth)
        chanDim = -1

        #change it if keras considers otherwise
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        inputs = Input(shape=inputShape)
        """        ^ the input of the network is actually the first "computation" in the graph"""

        if aug == True:
            inputs = augment.RandomFlip(mode="horizontal")(inputs)
            inputs = augment.RandomRotation(factor=0.1)(inputs)
            inputs = augment.RandomTranslation(height_factor=0.1, width_factor=0.1)(inputs)
            inputs = augment.RandomZoom(height_factor=0.1, width_factor=0.1)(inputs)
            inputs = augment.RandomContrast(factor=0.2)(inputs)

        x = GoogLeNet_.conv_module(inputs, 96, 3, 3, (1, 1), chanDim)

        x = GoogLeNet_.inception_module(x, 32, 0, 32, 0, 0, 0, chanDim)
        x = GoogLeNet_.inception_module(x, 32, 0, 48, 0, 0, 0, chanDim)
        x = GoogLeNet_.downsample_module(x, 80, chanDim)

        """4 inception modules alternating between more 1x1 and more 3x3 filters (as written in the
        whitepaper of the googlenet network"""
        x = GoogLeNet_.inception_module(x, 112, 0, 48, 0, 0, 0, chanDim)
        x = GoogLeNet_.inception_module(x, 96, 0, 64, 0, 0, 0, chanDim)
        x = GoogLeNet_.inception_module(x, 80, 0, 80, 0, 0, 0, chanDim)
        x = GoogLeNet_.inception_module(x, 48, 0, 96, 0, 0, 0, chanDim)
        x = GoogLeNet_.downsample_module(x, 80, chanDim)

        x = GoogLeNet_.inception_module(x, 176, 0, 160, 0, 0, 0, chanDim)
        x = GoogLeNet_.inception_module(x, 176, 0, 160, 0, 0, 0, chanDim)
        x = AveragePooling2D((7, 7))(x)
        x = Dropout(0.5)(x)

        x = Flatten()(x)
        x = Dense(num_classes)(x)
        x = Activation("softmax")(x)

        model = Model(inputs, x, name="minigooglenet")

        plot_model(model, to_file="outputs/MiniGoogLeNet.png", show_shapes=True)

        return model

    @staticmethod
    def build(width, height, depth, num_classes, aug=False, reg=0.0005):

        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        inputs = Input(shape=inputShape)

        if aug == True:
            x = augment.RandomFlip(mode="horizontal")(inputs)
            x = augment.RandomRotation(factor=0.1)(x)
            x = augment.RandomTranslation(height_factor=0.1, width_factor=0.1)(x)
            x = augment.RandomZoom(height_factor=0.1, width_factor=0.1)(x)
            x = augment.RandomContrast(factor=0.2)(x)
            x = GoogLeNet_.conv_module(x, 64, 5, 5, (1, 1), chanDim, reg=reg, name="block1")
        else:
            x = GoogLeNet_.conv_module(inputs, 64, 5, 5, (1, 1), chanDim, reg=reg, name="block1")

        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool1")(x)
        x = GoogLeNet_.conv_module(x, 64, 1, 1, (1, 1), chanDim, reg=reg, name="block2")
        x = GoogLeNet_.conv_module(x, 192, 3, 3, (1, 1), chanDim, reg=reg, name="block3")
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool2")(x)

        x = GoogLeNet_.inception_module(x, 64, 96, 128, 16, 32, 32, chanDim, "3a", reg)
        x = GoogLeNet_.inception_module(x, 128, 128, 192, 32, 96, 64, chanDim, "3b", reg)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool3")(x)

        x = GoogLeNet_.inception_module(x, 192, 96, 208, 16, 48, 64, chanDim, "4a", reg)
        x = GoogLeNet_.inception_module(x, 160, 112, 224, 24, 64, 64, chanDim, "4b", reg)
        x = GoogLeNet_.inception_module(x, 128, 128, 256, 24, 64, 64, chanDim, "4c", reg)
        x = GoogLeNet_.inception_module(x, 112, 144, 288, 32, 64, 64, chanDim, "4d", reg)
        x = GoogLeNet_.inception_module(x, 256, 160, 320, 32, 128, 128, chanDim, "4e", reg)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool4")(x)

        x = AveragePooling2D((4, 4), name="pool5")(x)
        x = Dropout(0.4, name="drop")(x)
        x = Flatten(name="flatten")(x)
        x = Dense(num_classes, kernel_regularizer=L2(reg), name="labels")(x)
        x = Activation("softmax", name="softmax")(x)

        model = Model(inputs, x, name="googlenet")

        plot_model(model, show_shapes=True, show_layer_names=True)

        return model


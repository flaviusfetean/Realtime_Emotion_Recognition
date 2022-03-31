from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers.experimental import preprocessing as augment
from tensorflow.keras import layers
import tensorflow.keras.backend as K

class SqueezeNet:
    @staticmethod
    def squeeze(input, numFilters, reg=0.0001, name=None):

        conv1x1 = layers.Conv2D(numFilters, kernel_size=(1, 1), strides=(1, 1),
                          kernel_regularizer=l2(reg), name=name + "_conv1x1")(input)
        act = layers.Activation("elu", name=name + "_elu")(conv1x1)

        return act

    @staticmethod
    def expand(input, numFilters, kernel_size=(1, 1), strides=(1, 1), reg=0.0001, name=None):

        conv_name = name + "_expand" + str(kernel_size[0]) + "x" + str(kernel_size[1])

        conv = layers.Conv2D(numFilters, kernel_size=kernel_size, strides=strides,
                             padding="same", kernel_regularizer=l2(reg), name=conv_name)(input)
        act = layers.Activation("elu", name=name + "_elu")(conv)

        return act

    @staticmethod
    def fire(input, numSqueezeFilters, numExpandFilters, reg=0.0001, chanDim=-1, name=None):

        squeeze1x1 = SqueezeNet.squeeze(input, numFilters=numSqueezeFilters,
                                       reg=reg, name=name + "_squeeze1x1")
        expand1x1 = SqueezeNet.expand(squeeze1x1, numExpandFilters, (1, 1), strides=(1, 1),
                                     reg=reg, name=name + "_expand1x1")
        expand3x3 = SqueezeNet.expand(squeeze1x1, numExpandFilters, (3, 3), strides=(1, 1),
                                     reg=reg, name=name + "_expand3x3")

        return layers.concatenate([expand1x1, expand3x3], axis=chanDim)

    @staticmethod
    def build(width, height, depth, numClasses, reg=0.0001, aug=False):

        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        input = layers.Input(inputShape)

        if aug == True:
            x = augment.RandomFlip(mode="horizontal")(input)
            x = augment.RandomRotation(factor=0.1)(x)
            x = augment.RandomTranslation(height_factor=0.1, width_factor=0.1)(x)
            x = augment.RandomZoom(height_factor=0.1, width_factor=0.1)(x)
            x = augment.RandomContrast(factor=0.2)(x)
            x = layers.Conv2D(64, kernel_size=(5, 5), strides=(1, 1), kernel_regularizer=l2(reg),
                              name="conv7x7_1")(x)
        else:
            x = layers.Conv2D(96, kernel_size=(5, 5), strides=(1, 1), kernel_regularizer=l2(reg),
                              name="conv7x7_1")(input)

        x = layers.Activation("elu", name="elu_1")(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool_1")(x)

        # 2*FIRE => POOL
        fire_2 = SqueezeNet.fire(x, 16, 64, reg, chanDim, name="fire_2")
        fire_3 = SqueezeNet.fire(fire_2, 16, 64, reg, chanDim, name="fire_3")
        pool_3 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool_3")(fire_3)

        # 3*FIRE => POOL
        fire_4 = SqueezeNet.fire(pool_3, 16, 64, reg, chanDim, name="fire_4")
        fire_5 = SqueezeNet.fire(fire_4, 32, 128, reg, chanDim, name="fire_5")
        pool_6 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="maxpool_6")(fire_5)

        # FIRE => DROPOUT => CONV => ELU => POOL => SOFTMAX(OUTPUT)
        fire_7 = SqueezeNet.fire(pool_6, 32, 128, reg, chanDim, name="fire_7")
        drop_7 = layers.Dropout(0.5, name="dropout_9")(fire_7)
        conv_8 = layers.Conv2D(numClasses, (1, 1), (1, 1), kernel_regularizer=l2(reg),
                                name="conv1x1_10")(drop_7)
        pool_8 = layers.AveragePooling2D(pool_size=(6, 6))(conv_8)
        flatten = layers.Flatten(name="flatten")(pool_8)
        softmax = layers.Activation("softmax", name="softmax_final")(flatten)

        model = Model(input, softmax, name="SqueezeNet")

        plot_model(model, "outputs/squeezenet.png", show_shapes=True)

        return model

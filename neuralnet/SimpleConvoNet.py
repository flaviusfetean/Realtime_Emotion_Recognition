from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as L
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow.keras.layers.experimental.preprocessing as aug

class Categorical_ConvoNet:
    @staticmethod


    def build(height, width, depth, layers, sizes, activations,
              alpha_reg_intensity=0, augment_data=False, initializer="glorot_uniform"):
        alpha = alpha_reg_intensity
        chanDim = -1
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            chanDim = 1
        else:
            input_shape = (height, width, depth)
            chanDim = -1

        model = Sequential()

        if augment_data == True:
            data_augmentation = tf.keras.Sequential([
                aug.RandomFlip(mode="horizontal"),
                aug.RandomRotation(factor=0.1),
                aug.RandomTranslation(height_factor=0.1, width_factor=0.1),
                aug.RandomZoom(height_factor=0.1, width_factor=0.1),
                aug.RandomContrast(factor=0.2)
            ])
            model.add(data_augmentation)

        for i in range(len(sizes)):
            size = sizes[i]
            layer = layers[i]
            activ = activations[i]

            if layer == "Conv2D" and len(size) == 2:
                size.append((1, 1))

            if i == 0:
                if layer == "Conv2D":
                    model.add(L.Conv2D(size[0], size[1], strides=size[2], padding="same",
                                       input_shape=input_shape, activation=activ,
                                       kernel_regularizer=tf.keras.regularizers.L2(alpha),
                                       kernel_initializer=initializer))
                elif layer == "Dense":
                    model.add(L.Dense(size, activation=activ, input_shape=input_shape,
                                      kernel_regularizer=tf.keras.regularizers.L2(alpha),
                                      kernel_initializer=initializer))
            else:
                if layer == "Conv2D":
                    model.add(L.Conv2D(size[0], size[1], strides=size[2], padding="same", activation=activ,
                                       kernel_regularizer=tf.keras.regularizers.L2(alpha),
                                       kernel_initializer=initializer))
                elif layer == "Pooling":
                    model.add(L.MaxPool2D(size[0], strides=size[1]))
                elif layer == "Dense":
                    if(layers[i - 1] != "Dense"):
                        model.add(L.Flatten())
                    model.add(L.Dense(size, activation=activ,
                                      kernel_regularizer=tf.keras.regularizers.L2(alpha),
                                      kernel_initializer=initializer))
                elif layer == "Batch Normalization":
                    model.add(L.BatchNormalization(axis=chanDim))
                elif layer == "Dropout":
                    model.add(L.Dropout(rate=size))
        return model
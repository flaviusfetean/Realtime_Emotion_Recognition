from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers.experimental import preprocessing as augment

class ResNet_:
    @staticmethod
    def residual_module(inputModel, num_filters, stride, chanDim, reduction=False,
                        reg=0.0001, bnEps=2e-5, bnMom=0.9):
        """

        :param inputModel: the model to which the residual module will be appended
        :param num_filters: the number of convolutional filters learnt
        :param stride: -
        :param chanDim: -
        :param reduction: specify if we want to perform dimensionality reduction => apply a conv layer with stride
        :param reg: -
        :param bnEps: = Batch Normalization Epsilon = small nr so we don t get ZeroDivisionError
        :param bnMom: = momentum applied to the bn layer
        :return: -
        """
        shortcut = inputModel

        #we apply bn first so it is "applied" to the input tensor
        bn1 = layers.BatchNormalization(axis=chanDim, epsilon=bnEps,
                                        momentum=bnMom)(inputModel)
        act1 = layers.Activation("relu")(bn1) # at act 1 we just have the input normalized
        #don't have a bias term in conv layer because the bn has one and we don't need two of them
        conv1 = layers.Conv2D(num_filters // 4, (1, 1), use_bias=False,
                              kernel_regularizer=l2(reg))(act1)

        bn2 = layers.BatchNormalization(axis=chanDim, epsilon=bnEps,
                                        momentum=bnMom)(conv1)
        act2 = layers.Activation("relu")(bn2)
        conv2 = layers.Conv2D(num_filters // 4, (3, 3), strides=stride,
                              padding="same", use_bias=False,
                              kernel_regularizer=l2(reg))(act2)

        bn3 = layers.BatchNormalization(axis=chanDim, epsilon=bnEps,
                                        momentum=bnMom)(conv2)
        act3 = layers.Activation("relu")(bn3)
        conv3 = layers.Conv2D(num_filters, (1, 1), use_bias=False,
                              kernel_regularizer=l2(reg))(act3)

        if reduction == True:
            shortcut = layers.Conv2D(num_filters, (1, 1), strides=stride,
                                     use_bias=False, kernel_regularizer=l2(reg))(act1)
                        # => this continuing act1 means applying a cov on itself normalized

        x = layers.add([conv3, shortcut])

        return x

    @staticmethod
    def build(width, height, depth, num_classes, stages, filters,
              aug=False, reg=0.0001, bnEps=2e-5, bnMom=0.9):
        """

        :param stages: stages are sequences of consecutive residual modules without reducing the size
                      -this parameter will receive a list where each index represents a stage(+1 because
                       of the array indexing: 0-first(=1), 2-second(=2), etc), and the
                      corresponding element means the number of residual modules stacked one after
                      another, for example: [5, 4, 6] - means we have 3 stages: first stage has 5
                      residual modules stacked, followed by a size reduction, then the second stage
                      has 4 residual modules stacked followed by a size reduction, etc.
        :param filters: refers to the filters of the residual modules. Receives a list with elements
                        representing the number of filters of the residual modules of ALL the modules
                        in the corresponding stage. This time index=stage(not +1), because we have an
                        additional first element that represents the num_filters for a Conv layer
                        applied directly to the input. Example: [64, 64, 128, 256] => input gets a
                        single convo layer with 64 filters, and then we have the whole first stage with
                        5 residual modules each training 64 filters, and then we have the whole second
                        stage with 4 modules training 128 filters each, etc.
        :return: -
        """
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        inputs = layers.Input(shape=inputShape)

        if aug == True:
            x = augment.RandomFlip(mode="horizontal")(inputs)
            x = augment.RandomRotation(factor=0.1)(x)
            x = augment.RandomTranslation(height_factor=0.1, width_factor=0.1)(x)
            x = augment.RandomZoom(height_factor=0.1, width_factor=0.1)(x)
            x = augment.RandomContrast(factor=0.2)(x)
            x = layers.BatchNormalization(axis=chanDim, epsilon=bnEps,
                                          momentum=bnMom)(x)
        else:
            x = layers.BatchNormalization(axis=chanDim, epsilon=bnEps,
                                          momentum=bnMom)(inputs)

        x = layers.Conv2D(filters[0], (5, 5), use_bias=False, padding="same",
                          kernel_regularizer=l2(reg))(x)
        x = layers.BatchNormalization(axis=chanDim, epsilon=bnEps,
                                      momentum=bnMom)(x)
        x = layers.Activation("relu")(x)
        x = layers.ZeroPadding2D((1, 1))(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        for i in range(0, len(stages)):
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet_.residual_module(x, filters[i + 1], stride, chanDim,
                        reduction=True, reg=reg, bnEps=bnEps, bnMom=bnMom)

            for j in range(0, stages[i] - 1):
                x = ResNet_.residual_module(x, filters[i + 1], (1, 1), chanDim,
                        reduction=False, reg=reg, bnEps=bnEps, bnMom=bnMom)

            x = layers.BatchNormalization(axis=chanDim, epsilon=bnEps,
                                          momentum=bnMom)(x)
            x = layers.Activation("relu")(x)
            x = layers.AveragePooling2D((8, 8))(x)

            x = layers.Flatten()(x)
            x = layers.Dense(num_classes, kernel_regularizer=l2(reg))(x)
            x = layers.Activation("softmax")(x)

            model = Model(inputs, x, name="resnet")

            plot_model(model, "outputs/resnet.png", show_shapes=True)

            return model

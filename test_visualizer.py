import visualkeras as vk
from tensorflow.keras import layers
from neuralnet.SimpleConvoNet import Categorical_ConvoNet as cnn
from tensorflow.keras.utils import plot_model
from collections import defaultdict

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
                   "elu", "none", "elu", "none", "softmax"]

    model = cnn.build(48, 48, 1, layers, sizes, activations, alpha_reg_intensity=0.0001,
                      augment_data=False)

    return model

model = build_mini_VGG()

#plot_model(model, show_shapes=True)

color_map = defaultdict(dict)

color_map[layers.Conv2D]['fill'] = 'red'
color_map[layers.MaxPooling2D]['fill'] = 'orange'
color_map[layers.Dense]['fill'] = color_map[layers.Flatten]['fill'] = 'green'
color_map[layers.BatchNormalization]['fill'] = 'white'
color_map[layers.Dropout]['fill'] = 'teal'

vk.layered_view(model, spacing=50, color_map=color_map, legend=True, scale_z=0.35, min_z=8).show()


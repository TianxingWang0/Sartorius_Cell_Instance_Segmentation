from tensorflow.keras import layers
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Sequential


def get_InceptionResNetV2(image_resize=(299, 299), weight_path=None):
    resnet = InceptionResNetV2(include_top=False, weights='imagenet', pooling='avg', input_shape=image_resize + (3,))
    model = Sequential()
    model.add(resnet)
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(3, activation='softmax'))
    model.layers[0].trainable = False
    if weight_path is not None:
        model.load_weights(weight_path)
    return model

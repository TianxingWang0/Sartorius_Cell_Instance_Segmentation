from utils import *
from glob import glob
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy


IMG_WIDTH = 300
IMG_HEIGHT = 300
IMG_DIM = (300,300)


"""

"""



def create_model():
    resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = resnet.output
    x = Flatten()(x)
    x = Dense(3, activation='softmax')(x)
    resnet.trainable = False
    model = Model(inputs=resnet.input, outputs=x)
    # model.summary()
    return model

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    zoom_range=0.1,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2,
    fill_mode='nearest')


train_generator = train_datagen.flow_from_directory(directory="/content/drive/My Drive/train_classifier",
                                                        shuffle=True,
                                                        batch_size=128, target_size=IMG_DIM, class_mode="categorical",
                                                        subset='training',
                                                        classes=("cort", "shsy5y","astro"))
val_generator = train_datagen.flow_from_directory(directory="/content/drive/My Drive/train_classifier",
                                                        batch_size=128, target_size=IMG_DIM, class_mode="categorical",
                                                        subset='validation',
                                                        classes=("cort", "shsy5y","astro"))


def train(lr,epochs):
    model = create_model();
    model.compile(loss=categorical_crossentropy,
                  optimizer=optimizers.Adam(lr=lr),
                  metrics=[categorical_accuracy])

    history = model.fit(train_generator,
                                      epochs=epochs,
                                      validation_data=val_generator,
                                      verbose=1)
    results = model.evaluate(val_generator)
    model.save('/content/drive/My Drive/576/resnet50_100epochslr0.00001.h5')






def get_classifier(weight_path):
    """

    :param weight_path: the path to the trained model weight
    :return: a loaded weight model
    """
    pass


def classify(image):
    """

    :param image: gray scale image with original size (520， 702）
    :return: the cell type contains in image : string "cort", "astro" or "shsy5y"
    """

    pass

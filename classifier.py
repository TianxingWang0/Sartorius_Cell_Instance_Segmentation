from utils import *
import os
import numpy as np
from glob import glob
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


df_train = pd.read_csv('train_processed2.csv')
X = df_train [['filename']]
Y = df_train [['cell_type']]
IMG_WIDTH = 704
IMG_HEIGHT = 520
IMG_DIM = (704,520)

# astro_files = glob('train_classifier/astro/*')
# cort_files = glob('train_classifier/cort/*')
# shsy5y_files = glob('train_classifier/shsy5y/*')
#
# allX = np.concatenate((astro_files,cort_files,shsy5y_files),axis=0)
# allY = [path.split('/')[1].split("\\")[0] for path in allX]
#
# print("astro: ", len(astro_files))
# print("cort: ", len(cort_files))
# print("shsy5y: ", len(shsy5y_files))



"""



"""
def create_model():

    resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = resnet.output
    x = Flatten()(x)
    x = Dense(3, activation='softmax')(x)
    model = Model(inputs=resnet.input, outputs=x)

    model.compile(loss=categorical_crossentropy,
                  optimizer=optimizers.Adam(lr=0.0001),
                  metrics=[categorical_accuracy])
    model.summary()
    return model

train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.1,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')
val_datagen = ImageDataGenerator(rescale=1./255)


#model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
model = create_model();
# evaluate using 10-fold cross validation
kfold = KFold(n_splits=10, shuffle=True, random_state=None)
for train_index, val_index in kfold.split(X,Y):
    training_data = df_train.iloc[train_index]
    validation_data = df_train.iloc[val_index]
    train_generator = train_datagen.flow_from_dataframe(training_data,directory ="train_classifier/train_data",
                                                        shuffle=True,
                                                        batch_size=5, target_size=IMG_DIM, class_mode="categorical",
                                                        x_col ="filename", y_col = "cell_type")
    val_generator = val_datagen.flow_from_dataframe(validation_data,directory ="train_classifier/train_data",
                                                    batch_size=5, target_size=IMG_DIM, class_mode="categorical",
                                                    x_col = "filename", y_col = "cell_type")


    history = model.fit_generator(train_generator,
                    steps_per_epoch=600,
                    epochs=5,
                    validation_data=val_generator,
                    validation_steps=108,
                    verbose=1)
    results = model.evaluate(val_generator)



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


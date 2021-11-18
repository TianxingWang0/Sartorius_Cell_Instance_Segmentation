"""

Data generator for model training

"""

from tensorflow import keras
import numpy as np
import cv2
import os
import random


def train_test_split(image_path, annotation_path, train_size, test_size):
    """
    for a train dataset with annotation, return the split train-test dataset, image and annotation file should be
    the same filename.
    Because annotation is less than image number, should load images who has annotation file
    :param image_path:
    :param annotation_path:
    :param train_size: the split train size
    :param test_size: the split test size, test_size + train_size is the total train data size
    :return: X_train, y_train, X_test, y_test
    """
    file_names = sorted([fname for fname in os.listdir(annotation_path) if fname.endswith('.png')])
    input_img_paths = [os.path.join(image_path, fname) for fname in file_names]
    annotation_img_paths = [os.path.join(annotation_path, fname) for fname in file_names]

    if train_size + test_size != len(input_img_paths):
        print("Warning! Train size and test size does not add up!")
    random.Random(1337).shuffle(input_img_paths)
    random.Random(1337).shuffle(annotation_img_paths)
    for input_path, target_path in zip(input_img_paths, annotation_img_paths):
        if input_path.split('/')[-1] != target_path.split('/')[-1]:
            print("Align Error!!!")
    return input_img_paths[test_size:], annotation_img_paths[test_size:], \
           input_img_paths[:test_size], annotation_img_paths[:test_size]


class DataGeneratorWithAnnotation(keras.utils.Sequence):
    """

    """

    def __init__(self, batch_size, img_size, input_img_paths, annotation_img_paths, reshape=False, preprocess=None):
        """

        :param batch_size:
        :param img_size: resize original image or not
        :param input_img_paths: align with annotation image order
        :param annotation_img_paths: align with input image order
        :param reshape: reshape to param img_size
        :param preprocess: user defined image preprocess
        """
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.annotation_img_paths = annotation_img_paths
        self.reshape = reshape
        self.preprocess = preprocess
        self.length = len(self.annotation_img_paths) // self.batch_size
        if len(self.annotation_img_paths) % self.batch_size > 0:
            self.length += 1

    def __len__(self):
        # return len(self.annotation_img_paths) // self.batch_size
        return self.length

    def __getitem__(self, idx):
        """
        Returns tuple (input, target) correspond to batch #idx.
        :param idx:
        :return: (x, y) batch of image and annotation image
        """
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.annotation_img_paths[i: i + self.batch_size]
        this_batch_size = len(batch_target_img_paths)
        x = np.zeros((this_batch_size,) + self.img_size + (1,), dtype="float32")
        if self.preprocess is None:
            if self.reshape:
                for j, path in enumerate(batch_input_img_paths):
                    x[j] = cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY), self.img_size).reshape(self.img_size + (1,))
            else:
                for j, path in enumerate(batch_input_img_paths):
                    x[j] = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY).reshape(self.img_size + (1,))
        else:  # has user define image preprocess
            if self.reshape:
                for j, path in enumerate(batch_input_img_paths):
                    x[j] = self.preprocess(
                        cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY), self.img_size)).reshape(
                        self.img_size + (1,))
            else:
                for j, path in enumerate(batch_input_img_paths):
                    x[j] = self.preprocess(
                        cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)).reshape(
                        self.img_size + (1,))
        x /= 255.0
        y = np.zeros((this_batch_size,) + self.img_size + (1,), dtype="uint8")
        if self.reshape:
            for j, path in enumerate(batch_target_img_paths):
                y[j] = cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY), self.img_size).reshape(
                       self.img_size + (1,))
        else:
            for j, path in enumerate(batch_target_img_paths):
                y[j] = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY).reshape(
                       self.img_size + (1,))
        return x, y

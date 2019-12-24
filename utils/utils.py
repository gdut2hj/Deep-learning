

import os
import random
import sys
import warnings
import cv2
import keras.backend as K
import numpy as np
import tensorflow as tf
from abc import ABCMeta, abstractmethod
from keras.preprocessing.image import apply_transform, flip_axis, random_channel_shift
from sklearn.utils import shuffle




class dataset1_Utils:
    '用于dataset1的预处理类'

    def __init__(self, input_height=224, input_width=224, output_height=224, output_width=224, shape=(224, 224), nClasses=12):
        self.dir_data = os.path.abspath('/dataset/dataset1')
        self.dir_seg = self.dir_data + "/annotations_prepped_train/"
        self.dir_img = self.dir_data + "/images_prepped_train/"
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.shape = shape
        self.n_classes = nClasses

    def getSegmentationArr(self, path, nClasses,  width, height):
        seg_labels = np.zeros((height, width, nClasses))
        img = cv2.imread(path, 1)
        img = cv2.resize(img, (width, height))
        img = img[:, :, 0]

        for c in range(nClasses):
            seg_labels[:, :, c] = (img == c).astype(int)
        # seg_labels = np.reshape(seg_labels, ( width*height,nClasses  ))
        return seg_labels

    def getImageArr(self, path, width, height):
        img = cv2.imread(path, 1)
        img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
        return img

    def readImgaeAndSeg(self):
        images = os.listdir(self.dir_img)
        images.sort()
        segmentations = os.listdir(self.dir_seg)
        segmentations.sort()
        X = []
        Y = []
        for im, seg in zip(images, segmentations):
            X.append(self.getImageArr(self.dir_img + im,
                                      self.input_width, self.input_height))
            Y.append(self.getSegmentationArr(self.dir_seg + seg,
                                             self.n_classes, self.output_width, self.output_height))

        X, Y = np.array(X), np.array(Y)
        print('X.shape,Y.shape: ', X.shape, Y.shape)
        return X, Y

    def splitDatasets(self, X, Y, train_rate=0.85, seed=None):
        np.random.seed(seed)
        index_train = np.random.choice(X.shape[0], int(
            X.shape[0]*train_rate), replace=False)
        index_test = list(set(range(X.shape[0])) - set(index_train))

        X, Y = shuffle(X, Y)
        X_train, y_train = X[index_train], Y[index_train]
        X_test, y_test = X[index_test], Y[index_test]

        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)
        return (X_train, y_train), (X_test, y_test)


class commonUtils:
    @staticmethod
    def GPUConfig(gpu_memory_fraction=0.90, gpu_device="1"):
        warnings.filterwarnings("ignore")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        config.gpu_options.visible_device_list = gpu_device
        K.set_session(tf.Session(config=config))
        print("python {}".format(sys.version))
        print("tensorflow version {}".format(tf.__version__))

    @staticmethod
    def IoU(Yi, y_predi):
        # mean Intersection over Union
        # Mean IoU = TP/(FN + TP + FP)

        IoUs = []
        Nclass = int(np.max(Yi)) + 1
        for c in range(Nclass):
            TP = np.sum((Yi == c) & (y_predi == c))
            FP = np.sum((Yi != c) & (y_predi == c))
            FN = np.sum((Yi == c) & (y_predi != c))
            IoU = TP/float(TP + FP + FN)
            print("class {:02.0f}: #TP={:6.0f}, #FP={:6.0f}, #FN={:5.0f}, IoU={:4.3f}".format(
                c, TP, FP, FN, IoU))
            IoUs.append(IoU)
        mIoU = np.mean(IoUs)
        print("_________________")
        print("Mean IoU: {:4.3f}".format(mIoU))




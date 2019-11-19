import numpy as np
import os
import cv2
import sys
import warnings
import tensorflow as tf
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from tensorflow.python.keras import backend as K


class VOC2012_Utils:
    pass


class dataset1_Utils:
    '用于dataset1的预处理类'

    def __init__(self):
        self.dir_data = '/home/ye/zhouhua/datasets/dataset1'
        self.dir_seg = dir_data + "/annotations_prepped_train/"
        self.dir_img = dir_data + "/images_prepped_train/"
        self.input_height = 224
        self.input_width = 224
        self.output_height = 224
        self.output_width = 224
        self.shape = (224, 224)
        self.n_classes = 12

    def getSegmentationArr(self, path, nClasses,  width, height):
        seg_labels = np.zeros((height, width, nClasses))
        img = cv2.imread(path, 1)
        img = cv2.resize(img, (width, height))
        img = img[:, :, 0]

        for c in range(nClasses):
            seg_labels[:, :, c] = (img == c).astype(int)
        ##seg_labels = np.reshape(seg_labels, ( width*height,nClasses  ))
        return seg_labels

    def getImageArr(self, path, width, height):
        img = cv2.imread(path, 1)
        img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
        return img

    def readImgaeAndSeg(self):
        images = os.listdir(dir_img)
        images.sort()
        segmentations = os.listdir(dir_seg)
        segmentations.sort()
        X = []
        Y = []
        for im, seg in zip(images, segmentations):
            X.append(getImageArr(dir_img + im, input_width, input_height))
            Y.append(getSegmentationArr(dir_seg + seg,
                                        n_classes, output_width, output_height))

        X, Y = np.array(X), np.array(Y)
        print('X.shape,Y.shape: ', X.shape, Y.shape)
        return X, Y

    def splitDatasets(self, X, Y, train_rate=0.85):
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

    def GPUConfig(self):
        warnings.filterwarnings("ignore")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        config.gpu_options.visible_device_list = "1"
        K.set_session(tf.Session(config=config))
        print("python {}".format(sys.version))
        print("tensorflow version {}".format(tf.__version__))

    def IoU(self, Yi, y_predi):
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

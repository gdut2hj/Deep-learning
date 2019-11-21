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

    def __init__(self, input_height=224, input_width=224, output_height=224, output_width=224, shape=(224, 224), nClasses=12):
        self.dir_data = '/home/ye/zhouhua/datasets/dataset1'
        self.dir_seg = self.dir_data + "/annotations_prepped_train/"
        self.dir_img = self.dir_data + "/images_prepped_train/"
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.shape = shape
        self.n_classes = nClasses
        pass

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
    @staticmethod
    def GPUConfig(gpu_memory_fraction=0.95, gpu_device="1"):
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


class dataset1_generator_reader:
    '用于dataset1的预处理类'

    def __init__(self, train_batch_size=16,
                 val_batch_size =16,
                 input_height=224,
                 input_width=224,
                 resize_height=224,
                 resize_width=224,
                 nClasses=12):
        self.dir_data = '/home/ye/zhouhua/datasets/dataset1'
        self.dir_seg = self.dir_data + "/annotations_prepped_train/"
        self.dir_img = self.dir_data + "/images_prepped_train/"
        self.input_height = input_height
        self.input_width = input_width
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.n_classes = nClasses
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_batch_index = 0
        self.val_batch_index = 0
        self.train_file_name_list = self.load_file_name_list(self.dir_img)
        self.val_file_name_list = self.load_file_name_list(self.dir_seg)
        self.n_train_file=len(self.train_file_name_list)  # 训练集的大小
        self.n_val_file=len(self.val_file_name_list)       # 验证集的大小
        self.n_train_steps_per_epoch = self.n_train_file // self.train_batch_size
        self.n_val_steps_per_epoch=self.n_val_file//self.val_batch_size  
         
    def load_file_name_list(self, filepath):
        images = os.listdir(filepath)
        images.sort()
        return images
    def next_train_batch(self, input_channel=3, output_channel=12):
        train_imgs=np.zeros((self.train_batch_size,self.resize_height,self.resize_width,input_channel))
        train_labels = np.zeros([self.train_batch_size, self.resize_height, self.resize_width, output_channel])
        if self.train_batch_index>=self.n_train_steps_per_epoch:
            # print("next train epoch")
            self.train_batch_index = 0
        #print('------------------')
        #print(self.train_batch_index)
        for i in range(self.train_batch_size):
            index=self.train_batch_size*self.train_batch_index+i
            #print('index'+str(index))
            img = Image.open(self.dir_img+self.train_file_name_list[index]) # 读取训练数据
            img=img.resize((self.resize_height,self.resize_width),Image.NEAREST)         # resize训练数据
            img = np.array(img)
            img = np.float32(img) / 127.5 - 1 # 归一化
            train_imgs[i]=img
            #print('train img shape is: ', img.shape)
            np.set_printoptions(threshold=np.inf)

            # label=Image.open(self.dir_seg+self.train_file_name_list[index])
            # label=label.resize((self.resize_height,self.resize_width),Image.NEAREST)
            # label=np.array(label, dtype=np.int32)
            #label[label == 255] = -1
            # label[label == 255] = 0
            #print('label>11:',label[label>11])  # 看看有没有大于11的label
            #print('label:', label)
            #print('label shape:',label.shape)
            # one_hot_label=self.make_one_hot(label,self.n_classes)
            label = self.getSegmentationArr(self.dir_seg+self.train_file_name_list[index], self.n_classes, self.resize_width, self.resize_height)
            train_labels[i]= label
            #print('one hot label shape:', one_hot_label.shape)
            #print('one hot label :', label)
            #print(label)

        self.train_batch_index+=1
        #print('------------------')


        return train_imgs, train_labels

    def next_val_batch(self, input_channel=3, output_channel=12):
        val_imgs = np.zeros((self.val_batch_size, self.resize_height, self.resize_width, input_channel))
        val_labels = np.zeros([self.val_batch_size, self.resize_height, self.resize_width, output_channel])
        if self.val_batch_index>=self.n_val_steps_per_epoch:
            #print("next train epoch")
            self.val_batch_index=0
        #print('------------------')
        #print(self.val_batch_index)


        for i in range(self.val_batch_size):
            index=self.val_batch_size*self.val_batch_index+i
            #print('index'+str(index))
            img=Image.open(self.dir_img+self.val_file_name_list[index])
            img = img.resize((self.resize_height, self.resize_width), Image.NEAREST)
            img = np.array(img)
            img = np.float32(img) / 127.5 - 1 # 归一化
            #print('val img shape is:', img.shape)
            val_imgs[i]=img

            #label = Image.open(self.dir_seg + self.val_file_name_list[index])
            label = self.getSegmentationArr(self.dir_seg + self.val_file_name_list[index], self.n_classes, self.resize_width, self.resize_height)
            # label[label == 255] = -1
            #label[label == 255] = 0
            #print(label[label>11])
            # print(label)
            # print(label.shape)
            #one_hot_label = self.make_one_hot(label, self.n_classes)
            val_labels[i]=label
        #print('------------------')
        self.val_batch_index += 1
        
        return val_imgs,val_labels

    def make_one_hot(self, x, n):
        one_hot = np.zeros([x.shape[0], x.shape[1], n])  # 256 256 21
        for i in range(x.shape[0]): # 256
            for j in range(x.shape[1]):  # 256
                one_hot[i,j,x[i,j]]=1
            return one_hot
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

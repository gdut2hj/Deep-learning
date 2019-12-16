import os
import random
import sys
import warnings

import cv2
import keras.backend as K
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.engine import Layer
from keras.preprocessing.image import (ImageDataGenerator, apply_transform,
                                       flip_axis, random_channel_shift)
from PIL import Image
from sklearn.utils import shuffle


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


class MaxPoolingWithArgmax2D(Layer):

    def __init__(
            self,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
            **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if K.backend() == 'tensorflow':
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = K.tf.nn.max_pool_with_argmax(
                inputs,
                ksize=ksize,
                strides=strides,
                padding=padding)
        else:
            errmsg = '{} backend is not supported for layer {}'.format(
                K.backend(), type(self).__name__)
            raise NotImplementedError(errmsg)
        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim // ratio[idx]
            if dim is not None else None
            for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(Layer):
    def __init__(self, up_size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.up_size = up_size

    def call(self, inputs, output_shape=None):

        updates, mask = inputs[0], inputs[1]
        with K.tf.variable_scope(self.name):
            mask = K.cast(mask, 'int32')
            input_shape = K.tf.shape(updates, out_type='int32')
            #  calculation new shape
            if output_shape is None:
                output_shape = (
                    input_shape[0],
                    input_shape[1] * self.up_size[0],
                    input_shape[2] * self.up_size[1],
                    input_shape[3])

            # calculation indices for batch, height, width and feature maps
            one_like_mask = K.ones_like(mask, dtype='int32')
            batch_shape = K.concatenate(
                [[input_shape[0]], [1], [1], [1]],
                axis=0)
            batch_range = K.reshape(
                K.tf.range(output_shape[0], dtype='int32'),
                shape=batch_shape)
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2]
            feature_range = K.tf.range(output_shape[3], dtype='int32')
            f = one_like_mask * feature_range

            # transpose indices & reshape update values to one dimension
            updates_size = K.tf.size(updates)
            indices = K.transpose(K.reshape(
                K.stack([b, y, x, f]),
                [4, updates_size]))
            values = K.reshape(updates, [updates_size])
            ret = K.tf.scatter_nd(indices, values, output_shape)
            return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
            mask_shape[0],
            mask_shape[1] * self.up_size[0],
            mask_shape[2] * self.up_size[1],
            mask_shape[3]
        )


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


class dataset1_generator_reader:
    '用于dataset1的预处理类'

    def __init__(self,
                images_data_dir='',
                 masks_data_dir='',
                 train_batch_size=16,
                 val_batch_size=16,
                 crop_size=(224, 224),
                 nClasses=12,
                 input_channel=3,
                 train_val_split_ratio=0.85,
                 seed=None
                 ):
        self.images_data_dir = images_data_dir
        self.masks_data_dir = masks_data_dir
        if images_data_dir == '' or masks_data_dir == '':
            raise ValueError('Invalid data:', images_data_dir, masks_data_dir,
                             'images or labels can not be empty!".')
        self.seed = seed
        self.crop_height = crop_size[0]
        self.crop_width = crop_size[1]
        self.n_classes = nClasses
        self.input_channel = input_channel
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_batch_index = 0
        self.val_batch_index = 0
        self.train_file_name_list, self.val_file_name_list = self.split_train_validation(
            self.images_data_dir, train_val_split_ratio,seed=self.seed)  # 得到训练集和验证集的文件名
        self.n_train_file = len(self.train_file_name_list)  # 训练集的大小
        self.n_val_file = len(self.val_file_name_list)  # 验证集的大小
        print('train and validation set size are: ',
              self.n_train_file, self.n_val_file)
        self.n_train_steps_per_epoch = self.n_train_file // self.train_batch_size
        self.n_val_steps_per_epoch = self.n_val_file//self.val_batch_size

    def train_generator_data(self):
        while True:
            # 返回VOC reader的next_train_batch()方法，参数为VOC_reader
            x, y = self.next_train_batch()
            #print('x.shape: y.shape:', x.shape, y.shape)
            yield (x, y)

    def val_generator_data(self):
        while True:
            x, y = self.next_val_batch()
            #print('val x.shape: val y.shape:', x.shape, y.shape)
            yield (x, y)

    def split_train_validation(self, filePath, split_ratio=0.85, shuffle=True, seed=None):
        if seed is not None:
            np.random.seed(seed)
        file_name_list = self.load_file_name_list(filePath)
        n_total = len(file_name_list)
        offset = int(n_total * split_ratio)
        assert n_total != 0 and offset > 0, 'split train validation error'
        if shuffle:
            random.shuffle(file_name_list)
        train_name_list = file_name_list[:offset]
        val_name_list = file_name_list[offset:]
        return train_name_list, val_name_list

    def load_file_name_list(self, filepath):
        images = os.listdir(filepath)
        images.sort()
        return images

    def next_train_batch(self):
        input_channel = self.input_channel
        output_channel = self.n_classes
        train_imgs = np.zeros(
            (self.train_batch_size, self.crop_height, self.crop_width, input_channel))
        train_labels = np.zeros(
            [self.train_batch_size, self.crop_height, self.crop_width, output_channel])
        if self.train_batch_index >= self.n_train_steps_per_epoch:
            # print("next train epoch")
            self.train_batch_index = 0
        # print('------------------')
        # print(self.train_batch_index)
        for i in range(self.train_batch_size):
            index = self.train_batch_size*self.train_batch_index+i
            img = Image.open(
                self.images_data_dir+self.train_file_name_list[index])  # 读取训练数据
            # img = img.resize((self.resize_height, self.resize_width),Image.NEAREST)  # resize训练数据
            img = np.array(img)
            label = self.getSegmentationArr(
                self.masks_data_dir+self.train_file_name_list[index], self.n_classes, self.crop_width, self.crop_height)
            img, label = self.pair_random_crop(
                img, label, (self.crop_height, self.crop_width), 'channels_last',sync_seed=self.seed)
            img = np.float32(img) / 127.5 - 1  # 归一化
            train_imgs[i] = img
            train_labels[i] = label
        self.train_batch_index += 1
        return train_imgs, train_labels

    def next_val_batch(self):
        input_channel = self.input_channel
        output_channel = self.n_classes
        val_imgs = np.zeros(
            (self.val_batch_size, self.crop_height, self.crop_width, input_channel))
        val_labels = np.zeros(
            [self.val_batch_size, self.crop_height, self.crop_width, output_channel])
        if self.val_batch_index >= self.n_val_steps_per_epoch:
            #print("next train epoch")
            self.val_batch_index = 0
        # print('------------------')
        # print(self.val_batch_index)

        for i in range(self.val_batch_size):
            index = self.val_batch_size*self.val_batch_index+i
            img = Image.open(self.images_data_dir+self.val_file_name_list[index])
            img = np.array(img)
            label = self.getSegmentationArr(
                self.masks_data_dir + self.val_file_name_list[index], self.n_classes, self.crop_width, self.crop_height)
            img, label = self.pair_random_crop(
                img, label, (self.crop_height, self.crop_width), 'channels_last',sync_seed=self.seed)
            img = np.float32(img) / 127.5 - 1  # 归一化
            val_imgs[i] = img
            val_labels[i] = label

        # print('------------------')
        self.val_batch_index += 1

        return val_imgs, val_labels

    def make_one_hot(self, x, n):
        one_hot = np.zeros([x.shape[0], x.shape[1], n])  # 256 256 21
        for i in range(x.shape[0]):  # 256
            for j in range(x.shape[1]):  # 256
                one_hot[i, j, x[i, j]] = 1
            return one_hot

    def pair_center_crop(self, x, y, center_crop_size, data_format, **kwargs):
        if data_format == 'channels_first':
            centerh, centerw = x.shape[1] // 2, x.shape[2] // 2
        elif data_format == 'channels_last':
            centerh, centerw = x.shape[0] // 2, x.shape[1] // 2   # 获取中心点
        # 获取剪裁后的中心点 channels_last
        lh, lw = center_crop_size[0] // 2, center_crop_size[1] // 2
        rh, rw = center_crop_size[0] - \
            lh, center_crop_size[1] - lw   # 得到剪裁后的中心点距离边缘的长度

        h_start, h_end = centerh - lh, centerh + rh
        # 计算原图从中心点开始裁剪的长和宽channels_last
        w_start, w_end = centerw - lw, centerw + rw
        if data_format == 'channels_first':
            return x[:, h_start:h_end, w_start:w_end], \
                y[:, h_start:h_end, w_start:w_end]
        elif data_format == 'channels_last':                        # 返回剪裁后的图像
            return x[h_start:h_end, w_start:w_end, :], \
                y[h_start:h_end, w_start:w_end, :]

    def random_transform(self, x, y, seed=None):
        # x is a single image, so it doesn't have image number at index 0
        img_row_index = 0   # we always use channels_last, so row index is 0, col is 1, channels is 2
        img_col_index = 1
        img_channel_index = 2
        if seed is not None:
            np.random.seed(seed)
        if self.crop_mode == 'none':
            crop_size = (x.shape[img_row_index], x.shape[img_col_index])
        else:
            crop_size = self.crop_size

        assert x.shape[img_row_index] == y.shape[img_row_index] and x.shape[img_col_index] == y.shape[
            img_col_index], 'DATA ERROR: Different shape of data and label!\ndata shape: %s, label shape: %s' % (str(x.shape), str(y.shape))

        # use composition of homographies to generate final transform that
        # needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * \
                np.random.uniform(-self.rotation_range,
                                  self.rotation_range)  # angle convert to radian
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],     # get rotation matrix
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if self.height_shift_range:
            # * x.shape[img_row_index]
            tx = np.random.uniform(-self.height_shift_range,
                                   self.height_shift_range) * crop_size[0]
        else:
            tx = 0

        if self.width_shift_range:
            # * x.shape[img_col_index]
            ty = np.random.uniform(-self.width_shift_range,
                                   self.width_shift_range) * crop_size[1]
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(
                self.zoom_range[0], self.zoom_range[1], 2)
        if self.zoom_maintain_shape:
            zy = zx
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transform_matrix = np.dot(
            np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

        h, w = x.shape[img_row_index], x.shape[img_col_index]

        transform_matrix = transform_matrix_offset_center(
            transform_matrix, h, w)

        x = apply_transform(x, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval)
        y = apply_transform(y, transform_matrix, img_channel_index,
                            fill_mode='constant', cval=self.label_cval)

        if self.channel_shift_range != 0:
            x = random_channel_shift(
                x, self.channel_shift_range, img_channel_index)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_index)
                y = flip_axis(y, img_col_index)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_index)
                y = flip_axis(y, img_row_index)

        if self.crop_mode == 'center':
            x, y = pair_center_crop(x, y, self.crop_size, self.data_format, seed=self.seed)
        elif self.crop_mode == 'random':
            x, y = pair_random_crop(x, y, self.crop_size, self.data_format, seed=self.seed)

        # TODO:
        # channel-wise normalization
        # barrel/fisheye
        return x, y

    def pair_random_crop(self, x, y, random_crop_size, data_format, sync_seed=None, **kwargs):
        if sync_seed is not None:
            np.random.seed(sync_seed)
        if data_format == 'channels_first':
            h, w = x.shape[1], x.shape[2]
        elif data_format == 'channels_last':
            h, w = x.shape[0], x.shape[1]           # get height and width
        # get difference from origin height and croped image height
        rangeh = (h - random_crop_size[0]) // 2
        rangew = (w - random_crop_size[1]) // 2
        offseth = 0 if rangeh == 0 else np.random.randint(
            rangeh)  # get random offset height and width
        offsetw = 0 if rangew == 0 else np.random.randint(rangew)

        h_start, h_end = offseth, offseth + \
            random_crop_size[0]     # height + random crop size
        w_start, w_end = offsetw, offsetw + \
            random_crop_size[1]  # width + random crop size

        if data_format == 'channels_first':
            return x[:, h_start:h_end, w_start:w_end], y[:, h_start:h_end, h_start:h_end]
        elif data_format == 'channels_last':
            return x[h_start:h_end, w_start:w_end, :], y[h_start:h_end, w_start:w_end, :]

    def standardize(self, x):
        pass

    def random_transform(self, x, y, crop_size, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # x is a single image, so it doesn't have image number at index 0
        img_row_index = self.row_index - 1
        img_col_index = self.col_index - 1
        img_channel_index = self.channel_index - 1

        assert x.shape[img_row_index] == y.shape[img_row_index] and x.shape[img_col_index] == y.shape[
            img_col_index], 'DATA ERROR: Different shape of data and label!\ndata shape: %s, label shape: %s' % (str(x.shape), str(y.shape))

        # use composition of homographies to generate final transform that
        # needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * \
                np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if self.height_shift_range:
            # * x.shape[img_row_index]
            tx = np.random.uniform(-self.height_shift_range,
                                   self.height_shift_range) * crop_size[0]
        else:
            tx = 0

        if self.width_shift_range:
            # * x.shape[img_col_index]
            ty = np.random.uniform(-self.width_shift_range,
                                   self.width_shift_range) * crop_size[1]
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(
                self.zoom_range[0], self.zoom_range[1], 2)
        if self.zoom_maintain_shape:
            zy = zx
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transform_matrix = np.dot(
            np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

        h, w = x.shape[img_row_index], x.shape[img_col_index]
        transform_matrix = transform_matrix_offset_center(
            transform_matrix, h, w)

        x = apply_transform(x, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval)
        y = apply_transform(y, transform_matrix, img_channel_index,
                            fill_mode='constant', cval=self.label_cval)

        if self.channel_shift_range != 0:
            x = random_channel_shift(
                x, self.channel_shift_range, img_channel_index)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_index)
                y = flip_axis(y, img_col_index)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_index)
                y = flip_axis(y, img_row_index)

        if self.crop_mode == 'center':
            x, y = pair_center_crop(x, y, self.crop_size, self.data_format,)
        elif self.crop_mode == 'random':
            x, y = pair_random_crop(x, y, self.crop_size, self.data_format,sync_seed=seed)

        # TODO:
        # channel-wise normalization
        # barrel/fisheye
        return x, y

    def getSegmentationArr(self, path, nClasses,  width, height):

        img = cv2.imread(path, 1)
        seg_labels = np.zeros((img.shape[0], img.shape[1], nClasses))
        #img = cv2.resize(img, (width, height))
        img = img[:, :, 0]

        for c in range(nClasses):
            seg_labels[:, :, c] = (img == c).astype(int)
        return seg_labels

'''
The implementation of base image preprocessing

Author: 
    zhouhua852
Project: 
    https://github.com/zhouhua852/Interesting-semantic-segmentation
Reference:
    keras 2.0.8: keras/preprocessing/images.py 
    github.com: https://github.com/aurora95/Keras-FCN/blob/master/utils/SegDataGenerator.py
'''

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


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


class Image_Data_Generator():
    'General image generator'

    def __init__(self,
                 preprocessing_function=None,
                 rescale=None,
                 samplewise_center=False,
                 samplewise_std_normalization=False,
                 crop_size=(224, 224),
                 crop_mode='none',
                 data_format=None,
                 rotation_range=0.,
                 height_shift_range=0.,
                 width_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 cval=0.,
                 label_cval=255.,
                 fill_mode='constant',
                 zoom_maintain_shape=True,
                 channel_shift_range=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 seed=None
                 ):
        if crop_mode not in {'none', 'random', 'center'}:               # 判断剪裁模式为随机还是中心剪裁
            raise Exception('crop_mode should be "none" or "random" or "center" '
                            'Received arg: ', crop_mode)

        if data_format is None:
            data_format = K.image_data_format()
        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError('`data_format` should be `"channels_last"` (channel after row and '
                             'column) or `"channels_first"` (channel before row and column). '
                             'Received arg: ', data_format)
        self.data_format = data_format
        if data_format == 'channels_first':
            self.channel_axis = 0
            self.row_axis = 1
            self.col_axis = 2
        if data_format == 'channels_last':
            self.channel_axis = 2
            self.row_axis = 0
            self.col_axis = 1

        if np.isscalar(zoom_range):         # 判断格式,是否是标量
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise Exception('zoom_range should be a float or '
                            'a tuple or list of two floats. '
                            'Received arg: ', zoom_range)

        self.fill_mode = fill_mode
        self.cval = cval
        self.label_cval = label_cval
        self.seed = seed
        self.preprocessing_function = preprocessing_function
        self.rescale = rescale
        self.samplewise_center = samplewise_center
        self.samplewise_std_normalization = samplewise_std_normalization
        self.crop_mode = crop_mode
        self.crop_size = crop_size
        self.rotation_range = rotation_range
        self.height_shift_range = height_shift_range
        self.width_shift_range = width_shift_range
        self.shear_range = shear_range
        self.zoom_maintain_shape = zoom_maintain_shape
        self.channel_shift_range = channel_shift_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

    def train_generator_data(self):
        while True:
            x, y = self.next_train_batch()
            # print('x.shape: y.shape:', x.shape, y.shape)
            yield (x, y)

    def val_generator_data(self):
        while True:
            x, y = self.next_val_batch()
            # print('val x.shape: val y.shape:', x.shape, y.shape)
            yield (x, y)

    def split_train_validation(self, filePath, split_ratio=0.85, shuffle=True, seed=None):
        '''
        split dataset to train and validationg dataset.

        # Arguments
        filePath: dataset filePath.
        split_ratio: train validation split ratio.
        shuffle: whether to shuffle file order.
        seed: numpy random seed.
        '''
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

    @abstractmethod
    def next_train_batch(self):
        pass

    @abstractmethod
    def next_val_batch(self):
        pass

    def make_one_hot(self, x, n):
        one_hot = np.zeros([x.shape[0], x.shape[1], n])  # 256 256 21
        for i in range(x.shape[0]):  # 256
            for j in range(x.shape[1]):  # 256
                one_hot[i, j, x[i, j]] = 1
            return one_hot

    def random_transform(self, x, y, seed=None):
        # x is a single image, so it doesn't have image number at index 0
        # we always use channels_last, so row index is 0, col is 1, channels is 2
        img_row_index = self.row_axis
        img_col_index = self.col_axis
        img_channel_index = self.channel_axis
        if seed is not None:
            np.random.seed(seed)
        if self.crop_mode == 'none':
            crop_size = (x.shape[img_row_index], x.shape[img_col_index])
        else:
            crop_size = self.crop_size

        # assert x.shape[img_row_index] == y.shape[img_row_index] and x.shape[img_col_index] == y.shape[
        #     img_col_index], 'DATA ERROR: Different shape of data and label!\ndata shape: %s, label shape: %s' % (str(x.shape), str(y.shape))

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
            x, y = self.pair_center_crop(
                x, y, self.crop_size, self.data_format, seed=self.seed)
        elif self.crop_mode == 'random':
            x, y = self.pair_random_crop(
                x, y, self.crop_size, self.data_format, seed=self.seed)

        # TODO:
        # channel-wise normalization
        # barrel/fisheye
        return x, y

    def standardize(self, x):
        """Apply the normalization configuration to a batch of inputs.

        # Arguments
            x: batch of inputs to be normalized.

        # Returns
            The inputs, normalized.
        """
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.rescale:
            x *= self.rescale
        # x is a single image, so it doesn't have image number at index 0
        img_channel_axis = self.channel_axis - 1
        if self.samplewise_center:
            x -= np.mean(x, axis=img_channel_axis, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_axis, keepdims=True) + 1e-7)

        # if self.featurewise_center:
        #     if self.mean is not None:
        #         x -= self.mean
        #     else:
        #         warnings.warn('This ImageDataGenerator specifies '
        #                       '`featurewise_center`, but it hasn\'t'
        #                       'been fit on any training data. Fit it '
        #                       'first by calling `.fit(numpy_data)`.')
        # if self.featurewise_std_normalization:
        #     if self.std is not None:
        #         x /= (self.std + 1e-7)
        #     else:
        #         warnings.warn('This ImageDataGenerator specifies '
        #                       '`featurewise_std_normalization`, but it hasn\'t'
        #                       'been fit on any training data. Fit it '
        #                       'first by calling `.fit(numpy_data)`.')
        # if self.zca_whitening:
        #     if self.principal_components is not None:
        #         flatx = np.reshape(x, (-1, np.prod(x.shape[-3:])))
        #         whitex = np.dot(flatx, self.principal_components)
        #         x = np.reshape(whitex, x.shape)
        #     else:
        #         warnings.warn('This ImageDataGenerator specifies '
        #                       '`zca_whitening`, but it hasn\'t'
        #                       'been fit on any training data. Fit it '
        #                       'first by calling `.fit(numpy_data)`.')
        return x

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

    def getSegmentationArr(self, path, nClasses,  width, height):

        img = cv2.imread(path, 1)
        seg_labels = np.zeros((img.shape[0], img.shape[1], nClasses))
        # img = cv2.resize(img, (width, height))
        img = img[:, :, 0]

        for c in range(nClasses):
            seg_labels[:, :, c] = (img == c).astype(int)
        return seg_labels

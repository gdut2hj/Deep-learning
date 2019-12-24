'''
The implementation of PASCAL VOC2012 dataset preprocessing for semantic segmentation

Author: 
    zhouhua852
Project: 
    https://github.com/zhouhua852/Interesting-semantic-segmentation
Reference:
    https://github.com/panxiaobai/voc_keras/blob/master/voc_reader.py
    
'''

import numpy as np
import os
from PIL import Image
from utils.ImageDataGenerator import Image_Data_Generator


class VOC2012_Utils(Image_Data_Generator):
    '''
    data generator for VOC2012
    '''

    def __init__(self,
                 dataset_dir='',
                 train_batch_size=32,
                 val_batch_size=32,
                 nClasses=12,
                 preprocessing_function=None,
                 rescale=None,
                 samplewise_center=False,
                 samplewise_std_normalization=False,
                 crop_size=(224, 224),
                 crop_mode=None,
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
                 seed=None):

        super(VOC2012_Utils, self).__init__(
            preprocessing_function,
            rescale,
            samplewise_center,
            samplewise_std_normalization,
            crop_size,
            crop_mode,
            data_format,
            rotation_range,
            height_shift_range,
            width_shift_range,
            shear_range,
            zoom_range,
            cval,
            label_cval,
            fill_mode,
            zoom_maintain_shape,
            channel_shift_range,
            horizontal_flip,
            vertical_flip,
            seed)
        self.dataset_dir = dataset_dir
        train_txt_filePath = os.path.join(
            self.dataset_dir, 'ImageSets/Segmentation/train.txt')
        val_txt_filePath = os.path.join(
            self.dataset_dir, 'ImageSets/Segmentation/val.txt')
        self.seed = seed
        self.train_batch_index = 0
        self.val_batch_index = 0
        self.nClasses = nClasses
        self.train_file_name_list = self.get_file_list(train_txt_filePath)
        self.val_file_name_list = self.get_file_list(val_txt_filePath)
        self.n_train_file = len(self.train_file_name_list)
        self.n_val_file = len(self.val_file_name_list)
        print('train and validation set size are: ',
              self.n_train_file, self.n_val_file)
        self.n_train_steps_per_epoch = self.n_train_file // self.train_batch_size
        self.n_val_steps_per_epoch = self.n_val_file // self.val_batch_size

    def get_file_list(self, filePath):
        fp = open(filePath)
        lines = fp.readlines
        fp.close()
        return lines

    def next_train_batch(self):
        input_channel = self.input_channel
        output_channel = self.nClasses
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
            img = np.array(img)
            label = Image.open(self.images_data_dir +
                               self.train_file_name_list[index] + '.jpg')
            img, label = self.random_transform(img, label, self.seed)
            img = self.standardize(img)
            label[label == 255] = 0
            label = self.one_hot(label, output_channel)
            train_labels[i] = label
        self.train_batch_index += 1
        return train_imgs, train_labels

    def next_val_batch(self):
        input_channel = self.input_channel
        output_channel = self.nClasses
        val_imgs = np.zeros(
            (self.val_batch_size, self.crop_height, self.crop_width, input_channel))
        val_labels = np.zeros(
            [self.val_batch_size, self.crop_height, self.crop_width, output_channel])
        if self.val_batch_index >= self.n_val_steps_per_epoch:
            # print("next train epoch")
            self.val_batch_index = 0
        # print('------------------')
        # print(self.val_batch_index)

        for i in range(self.val_batch_size):
            index = self.val_batch_size*self.val_batch_index+i
            img = Image.open(self.images_data_dir +
                             self.val_file_name_list[index])
            img = np.array(img)
            label = Image.open(self.images_data_dir +
                               self.val_file_name_list[index] + '.png')
            # do nothing to validation dataset
            img, label = self.random_transform(img, label, self.seed)
            img = self.standardize(img)
            label[label == 255] = 0
            label = self.one_hot(label, output_channel)

            val_imgs[i] = img
            val_labels[i] = label

        # print('------------------')
        self.val_batch_index += 1

        return val_imgs, val_labels

    def one_hot(self, label, num_classes):
        if np.ndim(label) == 3:
        label = np.squeeze(label, axis=-1)
        assert np.ndim(label) == 2

        heat_map = np.ones(shape=label.shape[0:2] + (num_classes,))
        for i in range(num_classes):
            heat_map[:, :, i] = np.equal(label, i).astype('float32')
        return heat_map

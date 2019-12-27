'''
The implementation of dataset1 preprocessing for semantic segmentation

Author: 
    zhouhua852
Project: 
    https://github.com/zhouhua852/Interesting-semantic-segmentation
Reference:
    https://github.com/panxiaobai/voc_keras/blob/master/voc_reader.py
    
'''
import numpy as np
from PIL import Image
from utils.ImageDataGenerator import Image_Data_Generator


class dataset1_generator_reader(Image_Data_Generator):
    '''
    data generator for dataset1
    '''

    def __init__(self,
                 images_data_dir='',
                 masks_data_dir='',
                 train_batch_size=16,
                 val_batch_size=16,
                 nClasses=12,
                 input_channel=3,
                 train_val_split_ratio=0.85,
                 crop_size=(224, 224)
                 ):
        super(dataset1_generator_reader, self).__init__()

        self.images_data_dir = images_data_dir
        self.masks_data_dir = masks_data_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.n_classes = nClasses
        self.input_channel = input_channel
        self.crop_height = crop_size[0]
        self.crop_width = crop_size[1]
        self.train_batch_index = 0
        self.val_batch_index = 0
        self.train_file_name_list, self.val_file_name_list = self.split_train_validation(
            self.images_data_dir, train_val_split_ratio, seed=self.seed)
        self.n_train_file = len(self.train_file_name_list)  # 训练集的大小
        self.n_val_file = len(self.val_file_name_list)  # 验证集的大小
        print('train and validation set size are: ',
              self.n_train_file, self.n_val_file)
        self.n_train_steps_per_epoch = self.n_train_file // self.train_batch_size
        self.n_val_steps_per_epoch = self.n_val_file // self.val_batch_size

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
                img, label, (self.crop_height, self.crop_width), 'channels_last', sync_seed=self.seed)
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
            # print("next train epoch")
            self.val_batch_index = 0
        # print('------------------')
        # print(self.val_batch_index)

        for i in range(self.val_batch_size):
            index = self.val_batch_size*self.val_batch_index+i
            img = Image.open(self.images_data_dir +
                             self.val_file_name_list[index])
            img = np.array(img)
            label = self.getSegmentationArr(
                self.masks_data_dir + self.val_file_name_list[index], self.n_classes, self.crop_width, self.crop_height)
            img, label = self.pair_random_crop(
                img, label, (self.crop_height, self.crop_width), 'channels_last', sync_seed=self.seed)
            img = np.float32(img) / 127.5 - 1  # 归一化
            val_imgs[i] = img
            val_labels[i] = label

        # print('------------------')
        self.val_batch_index += 1

        return val_imgs, val_labels

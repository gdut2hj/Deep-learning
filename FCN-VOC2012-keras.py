import random
import sys
import time
import os
import cv2
import tensorflow as tf
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
from models.FCN import FCN8
from utils.utils import commonUtils
from utils.VOC2012Utils import VOC2012_Utils
from config import VGG_Weights_path
import pickle


if __name__ == '__main__':
    commonUtils.GPUConfig()
    dataset_dir = os.path.abspath('/dataset/VOCdevkit/VOC2012/')
    images_data_dir = os.path.join(dataset_dir, 'JPEGImages/')
    labels_data_dir = os.path.join(dataset_dir, 'SegmentationClass/')
    dataGen = VOC2012_Utils(dataset_dir=dataset_dir,
                            images_data_dir=images_data_dir,
                            labels_data_dir=labels_data_dir,
                            train_batch_size=32,
                            val_batch_size=32,
                            nClasses=21,
                            crop_size=(224, 224),
                            crop_mode='random',
                            data_format='channels_last',
                            zoom_range=[0.5, 2],
                            label_cval=0,
                            channel_shift_range=20.
                            )

    tensorboard = TensorBoard(
        log_dir='./logs/dataset1/FCN-VOC2012-{}'.format(time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())))
    model = FCN8(nClasses=21,
                 input_height=224,
                 input_width=224,
                 VGG_Weights_path=VGG_Weights_path)
    model.summary()

    sgd = optimizers.SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy']
                  )
    best_weights_filepath = './data/FCN-VOC2012-best_weights.hdf5'
    earlyStopping = EarlyStopping(
        monitor='val_loss', patience=30, verbose=2, mode='auto')
    saveBestModel = ModelCheckpoint(
        best_weights_filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10,
                                  verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=1E-5)
    hist1 = model.fit_generator(generator=dataGen.train_generator_data(),
                                steps_per_epoch=dataGen.n_train_steps_per_epoch,
                                validation_data=dataGen.val_generator_data(),
                                validation_steps=dataGen.n_val_steps_per_epoch,
                                epochs=500,
                                verbose=2,
                                callbacks=[tensorboard, earlyStopping, saveBestModel, reduce_lr])
    with open('./data/FCN-VOC2012.pickle', 'wb') as file_pi:
        pickle.dump(hist1.history, file_pi)

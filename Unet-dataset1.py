import random
import sys
import time
import os
import cv2
import tensorflow as tf
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras import optimizers
from models.Unet import Unet
from utils.utils import dataset1_Utils, commonUtils, dataset1_generator_reader
import pickle


if __name__ == '__main__':
    commonUtils.GPUConfig(gpu_device='1')
    dataset_dir = os.path.abspath('/dataset/dataset1')
    images_data_dir = os.path.join(dataset_dir, 'images_prepped_train/')
    masks_data_dir = os.path.join(dataset_dir, 'annotations_prepped_train/')
    #print(images_data_dir, masks_data_dir)
    dataGen = dataset1_generator_reader(
        images_data_dir=images_data_dir,
        masks_data_dir=masks_data_dir,
        train_batch_size=16,
        val_batch_size=16,
        crop_size=(224, 224),
        nClasses=12,
        train_val_split_ratio=0.85
    )
    steps_per_epoch = dataGen.n_train_file//dataGen.train_batch_size
    validation_steps = dataGen.n_val_file // dataGen.val_batch_size
    train_generator = dataGen.train_generator_data()
    validation_generator = dataGen.val_generator_data()

    tensorboard = TensorBoard(
        log_dir='./logs/dataset1/Unet-dataset1-original-{}'.format(time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())))

    model = Unet(input_size=(224, 224, 3), num_class=12)
    model.summary()

    sgd = optimizers.SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy']
                  )
    best_weights_filepath = './models/Unet-best_weights.hdf5'
    earlyStopping = EarlyStopping(
        monitor='val_loss', patience=15, verbose=1, mode='auto')
    saveBestModel = ModelCheckpoint(
        best_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    hist1 = model.fit_generator(generator=train_generator,
                                steps_per_epoch=steps_per_epoch,
                                epochs=500,
                                validation_data=validation_generator,
                                validation_steps=validation_steps,
                                verbose=1,
                                callbacks=[tensorboard, earlyStopping, saveBestModel])
    with open('./data/Unet-dataset1.pickle', 'wb') as file_pi:
        pickle.dump(hist1.history, file_pi)

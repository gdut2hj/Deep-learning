import random
import sys
import time
import os
import cv2
import tensorflow as tf
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras import optimizers
from models import Segnet
from utils.utils import dataset1_Utils, commonUtils, dataset1_generator_reader
import pickle




if __name__ == '__main__':
    commonUtils.GPUConfig(gpu_device='0')
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
        log_dir='./logs/dataset1/SegNet-dataset1-{}'.format(time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())))

    model = Segnet(12, 224, 224)
    model.summary()

    sgd = optimizers.SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy']
                  )
    best_weights_filepath = './models/SegNet-best_weights.hdf5'
    earlyStopping = EarlyStopping(
        monitor='val_loss', patience=30, verbose=1, mode='auto')
    saveBestModel = ModelCheckpoint(
        best_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    #hist1 = model.fit(X_train, y_train,
    #                  validation_data=(X_test, y_test),
    #                  batch_size=32, epochs=200, verbose=1, callbacks=[tensorboard, earlyStopping, saveBestModel])
    # reload best weights
    # model.load_weights(best_weights_filepath)
    hist1 = model.fit_generator(generator=train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=500,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    verbose=1,
    callbacks=[tensorboard, earlyStopping, saveBestModel])
    with open('./data/SegNet-dataset1.pickle', 'wb') as file_pi:
        pickle.dump(hist1.history, file_pi)

import random
import sys
import time
import os
import cv2
import tensorflow as tf

import keras.backend as K
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
from models import FCN8
from utils.utils import dataset1_Utils, commonUtils
from models.FCN import FCN8
from config import VGG_Weights_path
import pickle

if __name__ == '__main__':
    commonUtils.GPUConfig(gpu_device="1")           
    dataGen = dataset1_Utils()
    X, Y = dataGen.readImgaeAndSeg()
    (X_train, y_train), (X_test, y_test) = dataGen.splitDatasets(X, Y, 0.85)
    with open('./data/X_test.pickle', 'wb') as file_pi:
        pickle.dump(X_test, file_pi)
    with open('./data/y_test.pickle', 'wb') as file_pi:
        pickle.dump(y_test, file_pi)
    tensorboard = TensorBoard(
         log_dir='./logs/dataset1/FCN-dataset1-{}'.format(time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())))
    model = FCN8(nClasses=12,
                 input_height=224,
                 input_width=224,
                 VGG_Weights_path=VGG_Weights_path)
    model.summary()

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    sgd = optimizers.SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy']
                  )
                  
    best_weights_filepath = './models/FCN_best_weights.hdf5'
    earlyStopping=EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='auto')
    best_weights_filepath = './data/FCN_best_weights.hdf5'
    earlyStopping=EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

    saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    
        
    hist1 = model.fit(X_train, y_train,
                      validation_data=(X_test, y_test),
                      batch_size=32, epochs=300, verbose=1, callbacks=[tensorboard, earlyStopping, saveBestModel, reduce_lr])
    with open('./data/FCN-dataset1-keras.pickle', 'wb') as file_pi:
        pickle.dump(hist1.history, file_pi)
    

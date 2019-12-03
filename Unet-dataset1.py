'''
说明：使用Unet模型训练dataset1，效果比FCN效果差很多.
可能原因分析：
1。没有使用预训练模型
'''
import random
import sys
import time
import os
import cv2
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.python.keras import optimizers
from models import FCN8, Unet_1, unet_2
from utils.utils import dataset1_Utils, commonUtils, dataset1_generator_reader
import pickle


def train_generator_data(dataset1_generator_reader):
    while True:
        # 返回VOC reader的next_train_batch()方法，参数为VOC_reader
        x, y = dataset1_generator_reader.next_train_batch()
        #print('x.shape: y.shape:',x.shape,y.shape)
        yield (x, y)


def val_generator_data(dataset1_generator_reader):
    while True:
        x, y = dataset1_generator_reader.next_val_batch()
        #print('val x.shape: val y.shape:',x.shape,y.shape)
        yield (x, y)


if __name__ == '__main__':
    commonUtils.GPUConfig()
    dataGen = dataset1_generator_reader(train_batch_size=16,
                                        val_batch_size=16,
                                        input_height=224,
                                        input_width=224,
                                        resize_height=224,
                                        resize_width=224,
                                        nClasses=12)
    steps_per_epoch=dataGen.n_train_file//dataGen.train_batch_size  # 22
    validation_steps = dataGen.n_val_file // dataGen.val_batch_size  # 22
    x_train = train_generator_data(dataGen)
    y_train = val_generator_data(dataGen)

    tensorboard = TensorBoard(
        log_dir='./logs/dataset1/Unet-dataset1-original-{}'.format(int(time.time())))

    model = unet_2()
    model.summary()

    sgd = optimizers.SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy']
                  )
    best_weights_filepath = './models/Unet-best_weights.hdf5'
    earlyStopping = EarlyStopping(
        monitor='val_loss', patience=10, verbose=1, mode='auto')
    saveBestModel = ModelCheckpoint(
        best_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    #hist1 = model.fit(X_train, y_train,
    #                  validation_data=(X_test, y_test),
    #                  batch_size=32, epochs=200, verbose=1, callbacks=[tensorboard, earlyStopping, saveBestModel])
    # reload best weights
    # model.load_weights(best_weights_filepath)
    hist1 = model.fit_generator(generator=x_train,
    steps_per_epoch=steps_per_epoch,
    epochs=500,
    validation_data=y_train,
    validation_steps=validation_steps,
    verbose=1,
    callbacks=[tensorboard, earlyStopping, saveBestModel])
    with open('./data/Unet-dataset1.pickle', 'wb') as file_pi:
        pickle.dump(hist1.history, file_pi)

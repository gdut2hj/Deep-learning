'''
说明：采用数据集较小的dataset1数据集进行训练，了解参数的调整方法和模型的相关训练方法。
'''
import random
import sys
import time
import os
import cv2
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.python.keras import optimizers
from utils import dataset1_Utils, commonUtils
from models import FCN8

if __name__ == '__main__':
    
    
    dataGen = dataset1_Utils()
    X, Y = dataGen.readImgaeAndSeg()
    (X_train, y_train), (X_test, y_test) = dataGen.splitDatasets(X, Y, 0.85)

    # # 保存测试数据，在jupyter中导入使用模型预测，观察效果
    # pickle_out = open("X_test.pickle", "wb")
    # pickle.dump(X_test, pickle_out)
    # pickle_out.close()

    # pickle_out = open("y_test.pickle", "wb")
    # pickle.dump(y_test, pickle_out)
    # pickle_out.close()

    tensorboard = TensorBoard(
         log_dir='/home/ye/zhouhua/logs/FCN/FCN-dataset1-keras-{}'.format(int(time.time())))

    model = FCN8(nClasses=12,
                 input_height=224,
                 input_width=224)
    model.summary()

    sgd = optimizers.SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy']
                  )
    best_weights_filepath = './models/best_weights.hdf5'
    earlyStopping=EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    hist1 = model.fit(X_train, y_train,
                      validation_data=(X_test, y_test),
                      batch_size=32, epochs=200, verbose=1, callbacks=[tensorboard, earlyStopping, saveBestModel])
    #reload best weights
    #model.load_weights(best_weights_filepath)
    

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
from models import FCN8
from utils.utils import dataset1_Utils, commonUtils
import pickle

if __name__ == '__main__':
    commonUtils.GPUConfig()
    if (os.path.exists("./data/X.pickle") and os.path.exists("./data/Y.pickle")):
        X = pickle.load('./data/X.pickle')
        Y = pickle.load('./data/Y.pickle')
        
    else:
        dataGen = dataset1_Utils()
        X, Y = dataGen.readImgaeAndSeg()
        # # 保存测试数据，在jupyter中导入使用模型预测，观察效果
        pickle_out = open("./data/X.pickle", "wb")
        pickle.dump(X, pickle_out)
        pickle_out.close()

        pickle_out = open("./data/Y.pickle'", "wb")
        pickle.dump(Y, pickle_out)
        pickle_out.close()
    
    (X_train, y_train), (X_test, y_test) = dataGen.splitDatasets(X, Y, 0.85)
    
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
    

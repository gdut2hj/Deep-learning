import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
import time
import cv2
from PIL import Image
from keras.preprocessing.image import *
from keras.utils.np_utils import to_categorical
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
import keras.backend as K
import tensorflow as tf
from models.FCN import FCN8


def inference(model_name, weight_file, image_size, image_list, data_dir, label_dir, return_results=True, save_dir=None,
              label_suffix='.png',
              data_suffix='.jpg'):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # mean_value = np.array([104.00699, 116.66877, 122.67892])
    batch_shape = (1, ) + image_size + (3, )
    save_path = os.path.join(current_dir, 'data/'+model_name)
    model_path = os.path.join(save_path, "model.json")
    checkpoint_path = os.path.join(save_path, weight_file)
    # model_path = os.path.join(current_dir, 'model_weights/fcn_atrous/model_change.hdf5')
    # model = FCN_Resnet50_32s((480,480,3))

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    K.set_session(session)

    model = globals()[model_name](nClasses=21,
                                  input_height=224,
                                  input_width=224,)
    model.load_weights(checkpoint_path, by_name=True)

    model.summary()

    results = []
    total = 0
    for img_num in image_list:
        img_num = img_num.strip('\n')
        total += 1
        print('#%d: %s' % (total, img_num))
        image = Image.open('%s/%s%s' % (data_dir, img_num, data_suffix))
        image = img_to_array(image)  # , data_format='default')

        label = Image.open('%s/%s%s' % (label_dir, img_num, label_suffix))
        label_size = label.size

        img_h, img_w = image.shape[0:2]

        # long_side = max(img_h, img_w, image_size[0], image_size[1])
        pad_w = max(image_size[1] - img_w, 0)
        pad_h = max(image_size[0] - img_h, 0)
        image = np.lib.pad(image, ((pad_h/2, pad_h - pad_h/2), (pad_w/2,
                                                                pad_w - pad_w/2), (0, 0)), 'constant', constant_values=0.)
        # image -= mean_value
        '''img = array_to_img(image, 'channels_last', scale=False)
        img.show()
        exit()'''
        # image = cv2.resize(image, image_size)

        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        result = model.predict(image, batch_size=1)
        result = np.argmax(np.squeeze(result), axis=-1).astype(np.uint8)

        result_img = Image.fromarray(result, mode='P')
        result_img.palette = label.palette
        # result_img = result_img.resize(label_size, resample=Image.BILINEAR)
        result_img = result_img.crop(
            (pad_w/2, pad_h/2, pad_w/2+img_w, pad_h/2+img_h))
        # result_img.show(title='result')
        if return_results:
            results.append(result_img)
        if save_dir:
            result_img.save(os.path.join(save_dir, img_num + '.png'))
    return results


def calculate_iou(model_name, nb_classes, res_dir, label_dir, image_list):
    conf_m = zeros((nb_classes, nb_classes), dtype=float)
    total = 0
    # mean_acc = 0.
    for img_num in image_list:
        img_num = img_num.strip('\n')
        total += 1
        print('#%d: %s' % (total, img_num))
        pred = img_to_array(Image.open('%s/%s.png' %
                                       (res_dir, img_num))).astype(int)
        label = img_to_array(Image.open('%s/%s.png' %
                                        (label_dir, img_num))).astype(int)
        flat_pred = np.ravel(pred)
        flat_label = np.ravel(label)
        # acc = 0.
        for p, l in zip(flat_pred, flat_label):
            if l == 255:
                continue
            if l < nb_classes and p < nb_classes:
                conf_m[l, p] += 1
            else:
                print('Invalid entry encountered, skipping! Label: ', l,
                      ' Prediction: ', p, ' Img_num: ', img_num)

        #    if l==p:
        #        acc+=1
        #acc /= flat_pred.shape[0]
        #mean_acc += acc
    #mean_acc /= total
    # print 'mean acc: %f'%mean_acc
    I = np.diag(conf_m)
    U = np.sum(conf_m, axis=0) + np.sum(conf_m, axis=1) - I
    IOU = I/U
    meanIOU = np.mean(IOU)
    return conf_m, IOU, meanIOU


def evaluate(model_name, weight_file, image_size, nb_classes, batch_size, val_file_path, data_dir, label_dir,
             label_suffix='.png',
             data_suffix='.jpg'):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(current_dir, 'data/'+model_name+'_res/')
    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)
    fp = open(val_file_path)
    image_list = fp.readlines()
    fp.close()

    start_time = time.time()
    inference(model_name, weight_file, image_size, image_list, data_dir, label_dir, return_results=False, save_dir=save_dir,  # do not save predict images
              label_suffix=label_suffix, data_suffix=data_suffix)
    duration = time.time() - start_time
    print('{}s used to make predictions.\n'.format(duration))

    start_time = time.time()
    conf_m, IOU, meanIOU = calculate_iou(
        model_name, nb_classes, save_dir, label_dir, image_list)
    print('IOU: ')
    print(IOU)
    print('meanIOU: %f' % meanIOU)
    print('pixel acc: %f' % (np.sum(np.diag(conf_m))/np.sum(conf_m)))
    duration = time.time() - start_time
    print('{}s used to calculate IOU.\n'.format(duration))


if __name__ == '__main__':
    model_name = 'FCN8'
    weight_file = 'FCN-VOC2012-best_wights.hdf5'
    image_size = (224, 224)
    nb_classes = 21
    batch_size = 1
    dataset = 'VOC2012'
    if dataset == 'VOC2012':

        val_file_path = os.path.expanduser(
            '/dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt')
        data_dir = os.path.expanduser(
            '/dataset/VOCdevkit/VOC2012/JPEGImages')
        label_dir = os.path.expanduser(
            '/dataset/VOCdevkit/VOC2012/SegmentationClass')
        label_suffix = '.png'
        data_suffix = '.jpg'

    evaluate(model_name, weight_file, image_size, nb_classes, batch_size, val_file_path, data_dir, label_dir,
             label_suffix=label_suffix, data_suffix=data_suffix)

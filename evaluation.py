import matplotlib.pyplot as plt 
import pickle
from keras.models import load_model
import numpy as np
import seaborn as sns
import random
from PIL import Image
import cv2
import os

print(os.listdir('.'))
model_path = './data/FCN_best_weights.hdf5'
history_path = './data/FCN-dataset1-keras.pickle'
X_test_dir = '/dataset/dataset1/images_prepped_test/'
y_test_dir = '/dataset/dataset1/annotations_prepped_test/'
nClasses=12
crop_width = 224
crop_height = 224
input_channel = 3

def pair_random_crop(x, y, random_crop_size, data_format, sync_seed=None, **kwargs):
        np.random.seed(sync_seed)
        if data_format == 'channels_first':
            h, w = x.shape[1], x.shape[2]
        elif data_format == 'channels_last':
            h, w = x.shape[0], x.shape[1]           # get height and width
        # get difference from origin height and croped image height
        rangeh = (h - random_crop_size[0]) // 2
        rangew = (w - random_crop_size[1]) // 2
        offseth = 0 if rangeh == 0 else np.random.randint(
            rangeh)  # get random offset height and width
        offsetw = 0 if rangew == 0 else np.random.randint(rangew)

        h_start, h_end = offseth, offseth + \
            random_crop_size[0]     # height + random crop size
        w_start, w_end = offsetw, offsetw + \
            random_crop_size[1]  # width + random crop size

        if data_format == 'channels_first':
            return x[:, h_start:h_end, w_start:w_end], y[:, h_start:h_end, h_start:h_end]
        elif data_format == 'channels_last':
            return x[h_start:h_end, w_start:w_end, :], y[h_start:h_end, w_start:w_end, :]

def getSegmentationArr(path, nClasses,  width, height):
        
        img = cv2.imread(path, 1)
        seg_labels = np.zeros((img.shape[0], img.shape[1], nClasses))
        #img = cv2.resize(img, (width, height))
        img = img[:, :, 0]

        for c in range(nClasses):
            seg_labels[:, :, c] = (img == c).astype(int)
        return seg_labels

def getTestData(test_dir_img, test_dir_seg):
    images = os.listdir(test_dir_img)
    images.sort()
    segmentations  = os.listdir(test_dir_seg)
    segmentations.sort()
    X = []
    Y = []
    for im , seg in zip(images,segmentations) :
        img = cv2.imread(test_dir_img+im)  # 读取训练数据   
        img = np.array(img)
        # print(img.shape)
        label = getSegmentationArr(test_dir_seg+seg, nClasses,crop_height, crop_width)
        img, label = pair_random_crop(img, label, (crop_height, crop_width), 'channels_last')

        img = np.float32(img) / 127.5 - 1  # 归一化
        X.append(img)
        Y.append(label)

    X, Y = np.array(X) , np.array(Y)
    print(X.shape,Y.shape)
    return X, Y

    from models.PSPnet import Interp 
model = load_model(model_path)

pickle_in = open(history_path,"rb")
history = pickle.load(pickle_in)
X_test, y_test = getTestData(X_test_dir, y_test_dir)

print(X_test.shape, y_test.shape)



for key in ['loss', 'val_loss']:
    plt.plot(history[key],label=key)
plt.legend()
plt.savefig('./data/FCN-dataset1-loss_val_loss.png',bbox_inches = 'tight', dpi=300)

for key in ['acc', 'val_acc']:
    plt.plot(history[key],label=key)
plt.legend()
plt.savefig('./data/FCN-dataset1-acc_val_acc.png',bbox_inches = 'tight', dpi=300)


y_pred = model.predict(X_test)
print(y_pred.shape)
y_predi = np.argmax(y_pred, axis=3)
y_testi = np.argmax(y_test, axis=3)
print(y_predi.shape, y_testi.shape)

def IoU(Yi,y_predi):
    ## mean Intersection over Union
    ## Mean IoU = TP/(FN + TP + FP)

    IoUs = []
    Nclass = int(np.max(Yi)) + 1
    for c in range(Nclass):
        TP = np.sum( (Yi == c)&(y_predi==c) )
        FP = np.sum( (Yi != c)&(y_predi==c) )
        FN = np.sum( (Yi == c)&(y_predi != c)) 
        IoU = TP/float(TP + FP + FN)
        print("class {:02.0f}: #TP={:6.0f}, #FP={:6.0f}, #FN={:5.0f}, IoU={:4.3f}".format(c,TP,FP,FN,IoU))
        IoUs.append(IoU)
    mIoU = np.mean(IoUs)
    print("_________________")
    print("Mean IoU: {:4.3f}".format(mIoU))
#增加Iou函数.
IoU(y_testi,y_predi)

def give_color_to_seg_img(seg,n_classes):
    '''
    seg : (input_width,input_height,3)
    '''
    
    if len(seg.shape)==3:
        seg = seg[:,:,0]
    seg_img = np.zeros( (seg.shape[0],seg.shape[1],3) ).astype('float')
    colors = sns.color_palette("hls", n_classes)
    
    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:,:,0] += (segc*( colors[c][0] ))
        seg_img[:,:,1] += (segc*( colors[c][1] ))
        seg_img[:,:,2] += (segc*( colors[c][2] ))

    return(seg_img)
shape = (224,224)
n_classes= 12

for i in range(3):
    img_is  = (X_test[i] + 1)*(255.0/2)
    seg = y_predi[i]
    segtest = y_testi[i]

    fig = plt.figure(figsize=(10,30))    
    ax = fig.add_subplot(1,3,1)
    ax.imshow(img_is/255.0)
    ax.set_title("original")
    
    ax = fig.add_subplot(1,3,2)
    ax.imshow(give_color_to_seg_img(seg,n_classes))
    ax.set_title("predicted class")
    
    ax = fig.add_subplot(1,3,3)
    ax.imshow(give_color_to_seg_img(segtest,n_classes))
    ax.set_title("true class")
    fig.savefig('./data/FCN-dataset1-test-seg_example_{}.png'.format(i), bbox_inches = 'tight',dpi=300)
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

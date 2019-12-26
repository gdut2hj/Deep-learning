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

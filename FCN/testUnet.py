from keras_segmentation.models.unet import vgg_unet
import cv2
import os 

data_dir = os.path.expanduser('E:/src/dataset/VOCtrainval_11-May-2012/VOCdevkit/VOC2012')

def read_images(root, train=True):
    txt_fname = root + '/ImageSets/Segmentation/' + (
        'train.txt' if train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    n = len(images)
    data, label = [None] * n, [None] * n
    for i, fname in enumerate(images):
        data[i] = cv2.imread('%s/JPEGImages/%s.jpg' % (
            root, fname))
        label[i] = cv2.imread('%s/SegmentationClass/%s.png' % (
            root, fname))
    return data, label

model = vgg_unet(n_classes=21 ,  input_height=500, input_width=500  )

model.train(
    train_images =  "dataset1/images_prepped_train/",
    train_annotations = "dataset1/annotations_prepped_train/",
    checkpoints_path = "/tmp/vgg_unet_1" , epochs=5
)

out = model.predict_segmentation(
    inp="dataset1/images_prepped_test/0016E5_07965.png",
    out_fname="/tmp/out.png"
)

import matplotlib.pyplot as plt
plt.imshow(out)

# evaluating the model 
print(model.evaluate_segmentation( inp_images_dir="dataset1/images_prepped_test/"  , annotations_dir="dataset1/annotations_prepped_test/" ) )
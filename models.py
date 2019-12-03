

from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Add, Activation, Input, concatenate, Dropout
from tensorflow.python.keras.layers import UpSampling2D, LeakyReLU, MaxPool2D, BatchNormalization
import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam
import segmentation_models as sm


def FCN8(nClasses, input_height=224, input_width=224):
    VGG_Weights_path = '/home/ye/zhouhua/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # input_height and width must be devisible by 32 because maxpooling with filter size = (2,2) is operated 5 times,
    # which makes the input_height and width 2^5 = 32 times smaller
    assert input_height % 32 == 0
    assert input_width % 32 == 0
    IMAGE_ORDERING = "channels_last"

    img_input = Input(shape=(input_height, input_width, 3))  # Assume 224,224,3

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same',           # 224 224 64
               name='block1_conv1', data_format=IMAGE_ORDERING)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',           # 224 224 64
               name='block1_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool',        # 112 112 64
                     data_format=IMAGE_ORDERING)(x)
    f1 = x

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same',          # 112 112 128
               name='block2_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same',          # 112 112 128
               name='block2_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool',        # 56 56 128
                     data_format=IMAGE_ORDERING)(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same',              # 56 56 256
               name='block3_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',              # 56 56 256
               name='block3_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',              # 56 56 256
               name='block3_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool',            # 28 28 256
                     data_format=IMAGE_ORDERING)(x)
    pool3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same',              # 28 28 512
               name='block4_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',              # 28 28 512
               name='block4_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',              # 28 28 512
               name='block4_conv3', data_format=IMAGE_ORDERING)(x)
    pool4 = MaxPooling2D((2, 2), strides=(
        2, 2), name='block4_pool', data_format=IMAGE_ORDERING)(x)  # (None, 14, 14, 512)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same',              # 14 14 512
               name='block5_conv1', data_format=IMAGE_ORDERING)(pool4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',              # 14 14 512
               name='block5_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',              # 14 14 512
               name='block5_conv3', data_format=IMAGE_ORDERING)(x)
    pool5 = MaxPooling2D((2, 2), strides=(
        2, 2), name='block5_pool', data_format=IMAGE_ORDERING)(x)  # (None, 7, 7, 512)

    #x = Flatten(name='flatten')(x)
    #x = Dense(4096, activation='relu', name='fc1')(x)
    # <--> o = ( Conv2D( 4096 , ( 7 , 7 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)
    # assuming that the input_height = input_width = 224 as in VGG data

    #x = Dense(4096, activation='relu', name='fc2')(x)
    # <--> o = ( Conv2D( 4096 , ( 1 , 1 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)
    # assuming that the input_height = input_width = 224 as in VGG data

    #x = Dense(1000 , activation='softmax', name='predictions')(x)
    # <--> o = ( Conv2D( nClasses ,  ( 1 , 1 ) ,kernel_initializer='he_normal' , data_format=IMAGE_ORDERING))(o)
    # assuming that the input_height = input_width = 224 as in VGG data

    vgg = Model(img_input, pool5)
    # loading VGG weights for the encoder parts of FCN8
    vgg.load_weights(VGG_Weights_path)

    n = 4096
    o = (Conv2D(n, (7, 7), activation='relu', padding='same',           # (None, 7, 7, 4096)  使用7x7卷积核，替换CNN结构的全连接层
                name="conv6", data_format=IMAGE_ORDERING))(pool5)
    conv7 = (Conv2D(n, (1, 1), activation='relu', padding='same',       # (None, 7, 7, 4096)
                    name="conv7", data_format=IMAGE_ORDERING))(o)

    # 4 times upsamping for pool4 layer
    conv7_4 = Conv2DTranspose(nClasses, kernel_size=(4, 4),  strides=(  # 4*(7-1)+4  (None, 28, 28, 12)
        4, 4), use_bias=False, data_format=IMAGE_ORDERING)(conv7)
    ## (None, 224, 224, 10)
    # 2 times upsampling for pool411
    pool411 = (Conv2D(nClasses, (1, 1), activation='relu', padding='same',      # pool4:(None, 14, 14, 512)   pool411:(None, 14, 14, 512)
                      name="pool4_11", data_format=IMAGE_ORDERING))(pool4)
    pool411_2 = (Conv2DTranspose(nClasses, kernel_size=(2, 2),  strides=(       # pool4ll_2: (None, 28, 28, 12)
        2, 2), use_bias=False, data_format=IMAGE_ORDERING))(pool411)

    pool311 = (Conv2D(nClasses, (1, 1), activation='relu', padding='same',      # pool3:(None, 28, 28, 256)  pool3ll: (None, 28, 28, 12)
                      name="pool3_11", data_format=IMAGE_ORDERING))(pool3)
    # 这里就是论文中提到的FCN8，即pool3+2倍上采样的pool4+4倍上采样。
    # (None, 28, 28, 12) add ((None, 28, 28, 12) add (None, 28, 28, 12)
    o = Add(name="add")([pool411_2, pool311, conv7_4])
    o = Conv2DTranspose(nClasses, kernel_size=(8, 8),  strides=(                # 28*8 (None, 224, 224, 12)
        8, 8), use_bias=False, data_format=IMAGE_ORDERING)(o)
    o = (Activation('softmax'))(o)

    model = Model(img_input, o)

    return model

def unet_2(pretrained_weights=None, input_size=(224, 224, 3), num_class=12):
    # 模型来源: https://blog.csdn.net/LawenceRay/article/details/97391350
    '''
    实际效果比FCN好，正宗Unet
    '''
    inputs = Input(input_size)
    conv1 = Conv2D(64,3,activation=None,padding='same',kernel_initializer='he_normal')(inputs)
    conv1 = LeakyReLU(alpha=0.3)(conv1)
    conv1 = Conv2D(64,3,activation=None,padding='same',kernel_initializer='he_normal')(conv1)
    conv1 = LeakyReLU(alpha=0.3)(conv1)
    pool1 = MaxPool2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(128,3,activation=None,padding='same',kernel_initializer='he_normal')(pool1)
    conv2 = LeakyReLU(alpha=0.3)(conv2)
    conv2 = Conv2D(128,3,activation=None,padding='same',kernel_initializer='he_normal')(conv2)
    conv2 = LeakyReLU(alpha=0.3)(conv2)
    pool2 = MaxPool2D(pool_size=(2,2))(conv2)

    conv3 = Conv2D(256,3,activation=None,padding='same',kernel_initializer='he_normal')(pool2)
    conv3 = LeakyReLU(alpha=0.3)(conv3)
    conv3 = Conv2D(256,3,activation=None,padding='same',kernel_initializer='he_normal')(conv3)
    conv3 = LeakyReLU(alpha=0.3)(conv3)
    pool3 = MaxPool2D(pool_size=(2,2))(conv3)

    conv4 = Conv2D(512,3,activation=None,padding='same',kernel_initializer='he_normal')(pool3)
    conv4 = LeakyReLU(alpha=0.3)(conv4)
    conv4 = Conv2D(512,3,activation=None,padding='same',kernel_initializer='he_normal')(conv4)
    conv4 = LeakyReLU(alpha=0.3)(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPool2D(pool_size=(2,2))(drop4)

    conv5 = Conv2D(1024,3,activation=None,padding='same',kernel_initializer='he_normal')(pool4)
    conv5 = LeakyReLU(alpha=0.3)(conv5)
    conv5 = Conv2D(1024,3,activation=None,padding='same',kernel_initializer='he_normal')(conv5)
    conv5 = LeakyReLU(alpha=0.3)(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512,2,activation=None,padding='same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(drop5))
    up6 = LeakyReLU(alpha=0.3)(up6)
    merge6 = concatenate([drop4,up6],axis=3)
    conv6 = Conv2D(512,3,activation=None,padding='same',kernel_initializer='he_normal')(merge6)
    conv6 = LeakyReLU(alpha=0.3)(conv6)
    conv6 = Conv2D(512,3,activation=None,padding='same',kernel_initializer='he_normal')(conv6)
    conv6 = LeakyReLU(alpha=0.3)(conv6)

    up7 = Conv2D(256,3,activation=None,padding='same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv6))
    up7 = LeakyReLU(alpha=0.3)(up7)
    merge7 = concatenate([conv3,up7],axis=3)
    conv7 = Conv2D(256,3,activation=None,padding='same',kernel_initializer='he_normal')(merge7)
    conv7 = LeakyReLU(alpha=0.3)(conv7)
    conv7 = Conv2D(256,3,activation=None,padding='same',kernel_initializer='he_normal')(conv7)
    conv7 = LeakyReLU(alpha=0.3)(conv7)


    up8 = Conv2D(128,3,activation=None,padding='same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv7))
    up8 = LeakyReLU(alpha=0.3)(up8)
    merge8 = concatenate([conv2,up8],axis=3)
    conv8 = Conv2D(128,3,activation=None,padding='same',kernel_initializer='he_normal')(merge8)
    conv8 = LeakyReLU(alpha=0.3)(conv8)
    conv8 = Conv2D(128,3,activation=None,padding='same',kernel_initializer='he_normal')(conv8)
    conv8 = LeakyReLU(alpha=0.3)(conv8)

    up9 = Conv2D(64,3,activation=None,padding='same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv8))
    up9 = LeakyReLU(alpha=0.3)(up9)
    merge9 = concatenate([conv1,up9],axis=3)
    conv9 = Conv2D(64,3,activation=None,padding='same',kernel_initializer='he_normal')(merge9)
    conv9 = LeakyReLU(alpha=0.3)(conv9)
    conv9 = Conv2D(64,3,activation=None,padding='same',kernel_initializer='he_normal')(conv9)
    conv9 = LeakyReLU(alpha=0.3)(conv9)

    if num_class == 2:#判断是否多分类问题，二分类激活函数使用sigmoid，而多分割采用softmax。相应的损失函数的选择也各有不同。
        conv10 = Conv2D(1,1,activation='sigmoid')(conv9)
        loss_function = 'binary_crossentropy'
    else:
        conv10 = Conv2D(num_class,1,activation='softmax')(conv9)
        loss_function = 'categorical_crossentropy'

    # conv10 = Conv2D(3,1,activation='softamx')(conv9)

    model = Model(inputs,conv10)

    #model.compile(optimizer=Adam(lr = 1e-5),loss=loss_function,metrics=['accuracy'])
    model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model

    # Build model
def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation == True:
        x = BatchActivate(x)
    return x

def residual_block(blockInput, num_filters=16, batch_activate = False):
    x = BatchActivate(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x
def Unet_ResNet_model(input_width,input_height, start_neurons=16, DropoutRatio=0.5, nClasses=12):
    input_layer = Input(shape =(input_width, input_height, 3))
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(input_layer)
    conv1 = residual_block(conv1,start_neurons * 1)
    conv1 = residual_block(conv1,start_neurons * 1, True)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio/2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2,start_neurons * 2)
    conv2 = residual_block(conv2,start_neurons * 2, True)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3,start_neurons * 4)
    conv3 = residual_block(conv3,start_neurons * 4, True)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
    conv4 = residual_block(conv4,start_neurons * 8)
    conv4 = residual_block(conv4,start_neurons * 8, True)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm,start_neurons * 16)
    convm = residual_block(convm,start_neurons * 16, True)
    
    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)
    
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4,start_neurons * 8)
    uconv4 = residual_block(uconv4,start_neurons * 8, True)
    
    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    #deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
    uconv3 = concatenate([deconv3, conv3])    
    uconv3 = Dropout(DropoutRatio)(uconv3)
    
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3,start_neurons * 4)
    uconv3 = residual_block(uconv3,start_neurons * 4, True)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
        
    uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2,start_neurons * 2)
    uconv2 = residual_block(uconv2,start_neurons * 2, True)
    
    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    #deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    
    uconv1 = Dropout(DropoutRatio)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1,start_neurons * 1)
    uconv1 = residual_block(uconv1,start_neurons * 1, True)
    
    #uconv1 = Dropout(DropoutRatio/2)(uconv1)
    #output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    output_layer_noActi = Conv2D(nClasses, (1,1), padding="same", activation=None)(uconv1)
    output_layer =  Activation('sigmoid')(output_layer_noActi)
    
    model = Model(input_layer, output_layer)
    return model

def PSPnet()

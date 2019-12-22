from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Add, Activation, Input, concatenate, Dropout, MaxPool2D, LeakyReLU, UpSampling2D
from keras.models import Model


def Unet(pretrained_weights=None, input_size=(224, 224, 3), num_class=12):
    '''
    model from: https://blog.csdn.net/LawenceRay/article/details/97391350
    ## Arguments
        pretrained_weights: pretrained weights path, make model converge faster.
        input_size: tuple, channels last.  
        num_class: Integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution).
    '''
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation=None, padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = LeakyReLU(alpha=0.3)(conv1)
    conv1 = Conv2D(64, 3, activation=None, padding='same',
                   kernel_initializer='he_normal')(conv1)
    conv1 = LeakyReLU(alpha=0.3)(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation=None, padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = LeakyReLU(alpha=0.3)(conv2)
    conv2 = Conv2D(128, 3, activation=None, padding='same',
                   kernel_initializer='he_normal')(conv2)
    conv2 = LeakyReLU(alpha=0.3)(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation=None, padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = LeakyReLU(alpha=0.3)(conv3)
    conv3 = Conv2D(256, 3, activation=None, padding='same',
                   kernel_initializer='he_normal')(conv3)
    conv3 = LeakyReLU(alpha=0.3)(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation=None, padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = LeakyReLU(alpha=0.3)(conv4)
    conv4 = Conv2D(512, 3, activation=None, padding='same',
                   kernel_initializer='he_normal')(conv4)
    conv4 = LeakyReLU(alpha=0.3)(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPool2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation=None, padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = LeakyReLU(alpha=0.3)(conv5)
    conv5 = Conv2D(1024, 3, activation=None, padding='same',
                   kernel_initializer='he_normal')(conv5)
    conv5 = LeakyReLU(alpha=0.3)(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation=None, padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    up6 = LeakyReLU(alpha=0.3)(up6)
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation=None, padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = LeakyReLU(alpha=0.3)(conv6)
    conv6 = Conv2D(512, 3, activation=None, padding='same',
                   kernel_initializer='he_normal')(conv6)
    conv6 = LeakyReLU(alpha=0.3)(conv6)

    up7 = Conv2D(256, 3, activation=None, padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    up7 = LeakyReLU(alpha=0.3)(up7)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation=None, padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = LeakyReLU(alpha=0.3)(conv7)
    conv7 = Conv2D(256, 3, activation=None, padding='same',
                   kernel_initializer='he_normal')(conv7)
    conv7 = LeakyReLU(alpha=0.3)(conv7)

    up8 = Conv2D(128, 3, activation=None, padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    up8 = LeakyReLU(alpha=0.3)(up8)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation=None, padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = LeakyReLU(alpha=0.3)(conv8)
    conv8 = Conv2D(128, 3, activation=None, padding='same',
                   kernel_initializer='he_normal')(conv8)
    conv8 = LeakyReLU(alpha=0.3)(conv8)

    up9 = Conv2D(64, 3, activation=None, padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    up9 = LeakyReLU(alpha=0.3)(up9)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation=None, padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = LeakyReLU(alpha=0.3)(conv9)
    conv9 = Conv2D(64, 3, activation=None, padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = LeakyReLU(alpha=0.3)(conv9)

    if num_class == 2:
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        loss_function = 'binary_crossentropy'
    else:
        conv10 = Conv2D(num_class, 1, activation='softmax')(conv9)
        loss_function = 'categorical_crossentropy'
    model = Model(inputs, conv10)
    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model

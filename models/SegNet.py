from keras.layers import *
from keras.models import Model


class MaxPoolingWithArgmax2D(Layer):

    def __init__(
            self,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
            **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if K.backend() == 'tensorflow':
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = K.tf.nn.max_pool_with_argmax(
                inputs,
                ksize=ksize,
                strides=strides,
                padding=padding)
        else:
            errmsg = '{} backend is not supported for layer {}'.format(
                K.backend(), type(self).__name__)
            raise NotImplementedError(errmsg)
        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim // ratio[idx]
            if dim is not None else None
            for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(Layer):
    def __init__(self, up_size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.up_size = up_size

    def call(self, inputs, output_shape=None):

        updates, mask = inputs[0], inputs[1]
        with K.tf.variable_scope(self.name):
            mask = K.cast(mask, 'int32')
            input_shape = K.tf.shape(updates, out_type='int32')
            #  calculation new shape
            if output_shape is None:
                output_shape = (
                    input_shape[0],
                    input_shape[1] * self.up_size[0],
                    input_shape[2] * self.up_size[1],
                    input_shape[3])

            # calculation indices for batch, height, width and feature maps
            one_like_mask = K.ones_like(mask, dtype='int32')
            batch_shape = K.concatenate(
                [[input_shape[0]], [1], [1], [1]],
                axis=0)
            batch_range = K.reshape(
                K.tf.range(output_shape[0], dtype='int32'),
                shape=batch_shape)
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2]
            feature_range = K.tf.range(output_shape[3], dtype='int32')
            f = one_like_mask * feature_range

            # transpose indices & reshape update values to one dimension
            updates_size = K.tf.size(updates)
            indices = K.transpose(K.reshape(
                K.stack([b, y, x, f]),
                [4, updates_size]))
            values = K.reshape(updates, [updates_size])
            ret = K.tf.scatter_nd(indices, values, output_shape)
            return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
            mask_shape[0],
            mask_shape[1] * self.up_size[0],
            mask_shape[2] * self.up_size[1],
            mask_shape[3]
        )


def Segnet(nClasses, input_height, input_width, VGG_Weights_path):
    '''
    model from: https://github.com/lsh1994/keras-segmentation/blob/master/Models/SegNet.py

    ## Arguments
        nClasses: Integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution).
        input_height: Integer, input height of the model, must divide by 32.
        input_width: Integer, input width of the model, must divide by 32.
        VGG_Weights_path: VGG weights path, make model converge faster.
    '''
    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, 3))

    # Block 1
    x = Conv2D(64, (3, 3),
               activation='relu',
               padding='same',
               name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3),
               activation='relu',
               padding='same',
               name='block1_conv2')(x)
    x, mask_1 = MaxPoolingWithArgmax2D(name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3),
               activation='relu',
               padding='same',
               name='block2_conv1')(x)
    x = Conv2D(128, (3, 3),
               activation='relu',
               padding='same',
               name='block2_conv2')(x)
    x, mask_2 = MaxPoolingWithArgmax2D(name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv1')(x)
    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv2')(x)
    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv3')(x)
    x, mask_3 = MaxPoolingWithArgmax2D(name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv1')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv3')(x)

    x, mask_4 = MaxPoolingWithArgmax2D(name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv1')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv3')(x)
    x, mask_5 = MaxPoolingWithArgmax2D(name='block5_pool')(x)

    Vgg_streamlined = Model(inputs=img_input, outputs=x)

    # load vgg16 weights
    Vgg_streamlined.load_weights(VGG_Weights_path)

    # decoder
    unpool_1 = MaxUnpooling2D()([x, mask_5])
    y = Conv2D(512, (3, 3), padding="same")(unpool_1)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Conv2D(512, (3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Conv2D(512, (3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    unpool_2 = MaxUnpooling2D()([y, mask_4])
    y = Conv2D(512, (3, 3), padding="same")(unpool_2)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Conv2D(512, (3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Conv2D(256, (3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    unpool_3 = MaxUnpooling2D()([y, mask_3])
    y = Conv2D(256, (3, 3), padding="same")(unpool_3)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Conv2D(256, (3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Conv2D(128, (3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    unpool_4 = MaxUnpooling2D()([y, mask_2])
    y = Conv2D(128, (3, 3), padding="same")(unpool_4)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Conv2D(64, (3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    unpool_5 = MaxUnpooling2D()([y, mask_1])
    y = Conv2D(64, (3, 3), padding="same")(unpool_5)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    y = Conv2D(nClasses, (1, 1), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("sigmoid")(y)

    model = Model(inputs=img_input, outputs=y)
    return model

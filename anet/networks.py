from tensorflow.keras import backend as K
from tensorflow.keras import Input
from tensorflow.keras.layers import Layer, Activation, Dropout, Conv2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import MaxPooling2D, Dense, GlobalAveragePooling2D, Embedding
from tensorflow.keras.layers import add, subtract, multiply
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU, PReLU
from tensorflow.keras.layers import Concatenate, concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import mobilenet
import tensorflow as tf
import numpy as np

def upConv(x, **kwargs):
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(strides=(1, 1), kernel_size=(3, 3), padding='same', **kwargs)(x)
    return x

def transposeConv(x, **kwargs):
    x = Conv2DTranspose(strides=(2, 2), kernel_size=(4, 4), padding='same', **kwargs)(x)
    return x

def UnetGenerator(input_size=256, input_channels=1, target_channels=1, base_filter=64, deconv=transposeConv, control_nc=0):
    stride = 2
    # ctrl_layer = Input(shape=[1], name="unet_ctrl")
    # emb = Embedding(32, 8)(ctrl_layer)

    # -------------------------------
    # ENCODER
    # C64-C128-C256-C512-C512-C512-C512-C512
    # 1 layer block = Conv - BN - LeakyRelu
    # -------------------------------

    input_layer = Input(shape=[input_size, input_size, input_channels], name="unet_input")

    # 1 encoder C64
    # skip batchnorm on this layer on purpose (from paper)
    en_1b = Conv2D(filters=base_filter, kernel_size=(4, 4), padding='same', strides=(stride, stride))(input_layer)
    en_1 = LeakyReLU(alpha=0.2)(en_1b)

    # 2 encoder C128
    en_2 = Conv2D(filters=base_filter*2, kernel_size=(4, 4), padding='same', strides=(stride, stride), use_bias=False)(en_1)
    en_2b = BatchNormalization(name='gen_en_bn_2')(en_2)
    en_2 = LeakyReLU(alpha=0.2)(en_2b)

    # 3 encoder C256
    en_3 = Conv2D(filters=base_filter*4, kernel_size=(4, 4), padding='same', strides=(stride, stride), use_bias=False)(en_2)
    en_3b = BatchNormalization(name='gen_en_bn_3')(en_3)
    en_3 = LeakyReLU(alpha=0.2)(en_3b)

    # 4 encoder C512
    en_4 = Conv2D(filters=base_filter*8, kernel_size=(4, 4), padding='same', strides=(stride, stride), use_bias=False)(en_3)
    en_4b = BatchNormalization(name='gen_en_bn_4')(en_4)
    en_4 = LeakyReLU(alpha=0.2)(en_4b)

    # 5 encoder C512
    en_5 = Conv2D(filters=base_filter*8, kernel_size=(4, 4), padding='same', strides=(stride, stride), use_bias=False)(en_4)
    en_5b = BatchNormalization(name='gen_en_bn_5')(en_5)
    en_5 = LeakyReLU(alpha=0.2)(en_5b)

    # 6 encoder C512
    en_6 = Conv2D(filters=base_filter*8, kernel_size=(4, 4), padding='same', strides=(stride, stride), use_bias=False)(en_5)
    en_6b = BatchNormalization(name='gen_en_bn_6')(en_6)
    en_6 = LeakyReLU(alpha=0.2)(en_6b)

    # 7 encoder C512
    en_7 = Conv2D(filters=base_filter*8, kernel_size=(4, 4), padding='same', strides=(stride, stride), use_bias=False)(en_6)
    en_7b = BatchNormalization(name='gen_en_bn_7')(en_7)
    en_7 = LeakyReLU(alpha=0.2)(en_7b)

    # 8 encoder C512
    en_8 = Conv2D(filters=base_filter*8, kernel_size=(4, 4), padding='same', strides=(stride, stride), use_bias=False)(en_7)
    en_8b = BatchNormalization(name='gen_en_bn_8')(en_8)


    # -------------------------------
    # DECODER
    # CD512-CD512-CD512-C512-C256-C128-C64-C1
    # 1 layer block = Relu - Conv - Upsample - BN - DO
    # also adds skip connections (Concatenate(axis=3)). Takes input from previous layer matching encoder layer
    # -------------------------------
    # 1 decoder CD512 (decodes en_8)
    de_1 = Activation('relu')(en_8b)
    de_1 = deconv(de_1, filters=base_filter*8, use_bias=False)
    de_1 = BatchNormalization(name='gen_de_bn_1')(de_1)

    # 2 decoder CD1024 (decodes en_7)
    de_2 = Concatenate(axis=3)([de_1, en_7b])
    de_2 = Activation('relu')(de_2)

    de_2 = deconv(de_2, filters=base_filter*8, use_bias=False)
    de_2 = BatchNormalization(name='gen_de_bn_2')(de_2)
    de_2 = Dropout(rate=0.5)(de_2)

    # 3 decoder CD1024 (decodes en_6)
    de_3 = Concatenate(axis=3)([de_2, en_6b])
    de_3 = Activation('relu')(de_3)

    de_3 = deconv(de_3, filters=base_filter*8, use_bias=False)
    de_3 = BatchNormalization(name='gen_de_bn_3')(de_3)
    de_3 = Dropout(rate=0.5)(de_3)

    # 4 decoder CD1024 (decodes en_5)
    de_4 = Concatenate(axis=3)([de_3, en_5b])
    de_4 = Activation('relu')(de_4)

    de_4 = deconv(de_4, filters=base_filter*8, use_bias=False)
    de_4 = BatchNormalization(name='gen_de_bn_4')(de_4)
    de_4 = Dropout(rate=0.5)(de_4)

    # 5 decoder CD1024 (decodes en_4)
    de_5 = Concatenate(axis=3)([de_4, en_4b])
    de_5 = Activation('relu')(de_5)

    de_5 = deconv(de_5, filters=base_filter*4, use_bias=False)
    de_5 = BatchNormalization(name='gen_de_bn_5')(de_5)
    # de_5 = Dropout(rate=0.5)(de_5)


    # 6 decoder C512 (decodes en_3)
    de_6 = Concatenate(axis=3)([de_5, en_3b])
    de_6 = Activation('relu')(de_6)

    de_6 = deconv(de_6, filters=base_filter*2, use_bias=False)
    de_6 = BatchNormalization(name='gen_de_bn_6')(de_6)
    # de_6 = Dropout(rate=0.5)(de_6)

    # 7 decoder CD256 (decodes en_2)
    de_7 = Concatenate(axis=3)([de_6, en_2b])
    de_7 = Activation('relu')(de_7)
    de_7 = deconv(de_7, filters=base_filter, use_bias=False)
    de_7 = BatchNormalization(name='gen_de_bn_7')(de_7)
    # de_7 = Dropout(rate=0.5)(de_7)

    # After the last layer in the decoder, a convolution is applied
    # to map to the number of output channels (3 in general,
    # except in colorization, where it is 2), followed by a Tanh
    # function.
    de_8 = Concatenate(axis=3)([de_7, en_1b])
    de_8 = Activation('relu')(de_8)
    de_8 = deconv(de_8, filters=target_channels*4, use_bias=False)
    de_out = Conv2D(filters=target_channels, kernel_size=(1, 1), padding='same', strides=(1, 1))(de_8)
#     de_out = Activation('tanh')(de_out)

    output = Model(inputs=[input_layer], outputs=[de_out], name='unet_model')
    return output

def UpBlock(x, num_filter, kernel_size=8, stride=4, padding='same', activation=LeakyReLU):
    up_conv1 = Conv2DTranspose(filters=num_filter, kernel_size=kernel_size, strides=stride, padding=padding)
    up_conv2 = Conv2D(filters=num_filter, kernel_size=kernel_size, strides=stride, padding=padding)
    up_conv3 = Conv2DTranspose(filters=num_filter, kernel_size=kernel_size, strides=stride, padding=padding)
    h0 = activation()(up_conv1(x))
    l0 = activation()(up_conv2(h0))
    h1 = activation()(up_conv3(subtract([l0, x])))
    return add([h1, h0])

def D_UpBlock(x, num_filter, kernel_size=8, stride=4, padding='same', activation=LeakyReLU):
    conv = Conv2D(filters=num_filter, kernel_size=1, strides=1, padding='same')
    x = activation()(conv(x))
    return UpBlock(x, num_filter, kernel_size=kernel_size, stride=stride, padding=padding, activation=activation)


def DownBlock(x, num_filter, kernel_size=8, stride=4, padding=2, activation=LeakyReLU):
    down_conv1 = Conv2D(filters=num_filter, kernel_size=kernel_size, strides=stride, padding=padding)
    down_conv2 = Conv2DTranspose(filters=num_filter, kernel_size=kernel_size, strides=stride, padding=padding)
    down_conv3 = Conv2D(filters=num_filter, kernel_size=kernel_size, strides=stride, padding=padding)
    l0 = activation()(down_conv1(x))
    h0 = activation()(down_conv2(l0))
    l1 = activation()(down_conv3(subtract([h0, x])))
    return add([l1, l0])

def D_DownBlock(x, num_filter, kernel_size=8, stride=4, padding='same', activation=LeakyReLU):
    conv = Conv2D(filters=num_filter, kernel_size=1, strides=1, padding='same')
    x = activation()(conv(x))
    return DownBlock(x, num_filter, kernel_size=kernel_size, stride=stride, padding=padding, activation=activation)

def DBPNGenerator( input_size, input_channels, target_channels, scale_factor, base_filter=32, feat=128, depth=2, activation='leaky_relu', output_layer=None):
    if scale_factor == 2:
        kernel = 6
        stride = 2
    elif scale_factor == 4:
        kernel = 8
        stride = 4
    elif scale_factor == 8:
        kernel = 12
        stride = 8
    else:
        raise NotImplemented

    if activation == 'leaky_relu':
        activation = LeakyReLU
    elif activation == 'prelu':
        activation = PReLU
    elif type(activation) is str:
        activation = Activation(activation)

    padding = 'same'
    #Initial Feature Extraction
    input_ = Input(shape=[input_size, input_size, input_channels], name="dbpn_input")
    feat0 = Conv2D(filters=feat, kernel_size=(3, 3), padding=padding, strides=(1, 1))
    feat1 = Conv2D(filters=base_filter, kernel_size=(1, 1), padding=padding, strides=(1, 1))

    x = activation()(feat0(input_))
    x = activation()(feat1(x))

    #Back-projection stages
    concat_h = UpBlock(x, base_filter, kernel, stride, padding, activation)
    concat_l = DownBlock(concat_h, base_filter, kernel, stride, padding, activation)
    h = UpBlock(concat_l, base_filter, kernel, stride, padding, activation)

    for i in range(depth-1):
        concat_h = Concatenate(axis=3)([h, concat_h])
        l = D_DownBlock(concat_h, base_filter, kernel, stride, padding, activation)

        concat_l = Concatenate(axis=3)([l, concat_l])
        h = D_UpBlock(concat_l, base_filter, kernel, stride, padding, activation)

    concat_h = Concatenate(axis=3)([h, concat_h])
    if output_layer is None:
        #Reconstruction
        output_conv = Conv2D(filters=target_channels, kernel_size=(3, 3), padding='same', strides=(1, 1))
        x = output_conv(concat_h)
    else:
        x = output_layer(concat_h)
    #x = Activation('tanh')(x)
    output = Model(inputs=[input_], outputs=[x], name='dbpn_generator')
    return output


def get_dssim_l1_loss(alpha=0.84, kernel_size=3, max_value=1.0):
    from anet.dssim import DSSIMObjective
    from tensorflow.keras.losses import mean_absolute_error
    dssim = DSSIMObjective(kernel_size=kernel_size, max_value=max_value)
    def DSSIM_L1(y_true, y_pred):
        return alpha*dssim(y_true, y_pred) + (1.0-alpha)*mean_absolute_error(y_true, y_pred)
    return DSSIM_L1


def relu6(x):
    return K.relu(x, max_value=6)


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1), block_id=1):
    """Adds an initial convolution layer (with batch normalization and relu6).

    # Arguments
        inputs: Input tensor of shape `(rows, cols, 3)`
            (with `channels_last` data format) or
            (3, rows, cols) (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(224, 224, 3)` would be one valid value.
        filters: Integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.

    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv_%d' % block_id)(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_%d_relu' % block_id)(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    """Adds a depthwise convolution block.

    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.

    # Arguments
        inputs: Input tensor of shape `(rows, cols, channels)`
            (with `channels_last` data format) or
            (channels, rows, cols) (with `channels_first` data format).
        pointwise_conv_filters: Integer, the dimensionality of the output space
            (i.e. the number output of filters in the pointwise convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        block_id: Integer, a unique identification designating the block number.

    # Input shape
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.

    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)


def MobileUNet(input_size=256, input_channels=1, target_channels=1, base_filter=64, output_layer=None,
               alpha=1.0,
               alpha_up=1.0,
               depth_multiplier=1,
               dropout=1e-3):
    input_layer = Input(shape=[input_size, input_size, input_channels], name="mobile_unet_input")
    i00 = _depthwise_conv_block(input_layer, 8, alpha, depth_multiplier, block_id=100, strides=(1, 1))

    b00 = _depthwise_conv_block(i00, 64, alpha, depth_multiplier, block_id=0, strides=(2, 2)) # ==> 56
    b01 = _depthwise_conv_block(b00, 64, alpha, depth_multiplier, block_id=1) # ==> 28

    b02 = _depthwise_conv_block(b01, 128, alpha, depth_multiplier, block_id=2, strides=(2, 2)) # ==> 14
    b03 = _depthwise_conv_block(b02, 128, alpha, depth_multiplier, block_id=3)

    b04 = _depthwise_conv_block(b03, 256, alpha, depth_multiplier, block_id=4, strides=(2, 2))
    b05 = _depthwise_conv_block(b04, 256, alpha, depth_multiplier, block_id=5)

    b06 = _depthwise_conv_block(b05, 512, alpha, depth_multiplier, block_id=6, strides=(2, 2))
    b07 = _depthwise_conv_block(b06, 512, alpha, depth_multiplier, block_id=7)
    b08 = _depthwise_conv_block(b07, 512, alpha, depth_multiplier, block_id=8)
    b09 = _depthwise_conv_block(b08, 512, alpha, depth_multiplier, block_id=9)
    b10 = _depthwise_conv_block(b09, 512, alpha, depth_multiplier, block_id=10)
    b11 = _depthwise_conv_block(b10, 512, alpha, depth_multiplier, block_id=11)

    b12 = _depthwise_conv_block(b11, 1024, alpha, depth_multiplier, block_id=12, strides=(2, 2))
    b13 = _depthwise_conv_block(b12, 1024, alpha, depth_multiplier, block_id=13)
    # b13 = Dropout(dropout)(b13)

    filters = int(512 * alpha)
    up1 = concatenate([
        Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(b13),
        b11,
    ], axis=3)
    b14 = _depthwise_conv_block(up1, filters, alpha_up, depth_multiplier, block_id=14)

    filters = int(256 * alpha)
    up2 = concatenate([
        Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(b14),
        b05,
    ], axis=3)
    b15 = _depthwise_conv_block(up2, filters, alpha_up, depth_multiplier, block_id=15)

    filters = int(128 * alpha)
    up3 = concatenate([
        Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(b15),
        b03,
    ], axis=3)
    b16 = _depthwise_conv_block(up3, filters, alpha_up, depth_multiplier, block_id=16)

    filters = int(64 * alpha)
    up4 = concatenate([
        Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(b16),
        b01,
    ], axis=3)
    b17 = _depthwise_conv_block(up4, filters, alpha_up, depth_multiplier, block_id=17)

    filters = int(32 * alpha)
    up5 = concatenate([
        Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(b17),
        i00,
    ], axis=3)
    #b18 = _depthwise_conv_block(up5, filters, alpha_up, depth_multiplier, block_id=18)
    b18 = _conv_block(up5, filters, alpha_up, block_id=18)

    #x = Conv2D(target_channels, (1, 1), kernel_initializer='he_normal', activation='linear')(b18)

    # refinement layers similar to Deep Image Matting https://arxiv.org/pdf/1703.03872.pdf
    b19 = _depthwise_conv_block(b18, filters, alpha_up, depth_multiplier, block_id=19)
    b20 = add([b18, b19])

    
    x = Conv2D(target_channels, (1, 1), kernel_initializer='he_normal', activation='linear')(b20)
    # x = BilinearUpSampling2D(size=(2, 2))(x)
    # x = UpSampling2D(size=(2, 2))(x)
    # x = Activation('sigmoid')(x)
    if output_layer is not None:
        x = output_layer(x)

    model = Model(input_layer, x)

    return model


if __name__ == '__main__':
    model = UnetGenerator(input_size=256, input_channels=16, target_channels=1, base_filter=16)
    model = DBPNGenerator(input_size=256, input_channels=1, target_nc=1, scale_factor=4, base_filter=16, feat=64, depth=2)

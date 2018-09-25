from keras import backend as K
from keras.layers.core import Layer
from keras.layers import Layer, Activation, Input, Dropout, Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import MaxPooling2D, Dense, GlobalAveragePooling2D, Embedding
from keras.layers import add, subtract, multiply
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.merge import Concatenate
from keras.models import Sequential, Model
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
    from keras_contrib.losses import DSSIMObjective
    from keras.losses import mean_absolute_error
    dssim = DSSIMObjective(kernel_size=kernel_size, max_value=max_value)
    def DSSIM_L1(y_true, y_pred):
        return alpha*dssim(y_true, y_pred) + (1.0-alpha)*mean_absolute_error(y_true, y_pred)
    return DSSIM_L1

if __name__ == '__main__':
    model = UnetGenerator(input_size=256, input_channels=16, target_channels=1, base_filter=16)
    model = DBPNGenerator(input_size=256, input_channels=1, target_nc=1, scale_factor=4, base_filter=16, feat=64, depth=2)

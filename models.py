import tensorflow as tf
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.normalization import BatchNormalization

from model_layers import conv, global_average_pooling, max_pool, double_conv_res_block


def m34_res(x, num_classes=10):
    x = conv(x, pattern='[80/4, 64]')
    x = max_pool(x, pool_size=4)
    for i in range(3):
        x = double_conv_res_block(x, conv_pattern='[3, 48]')
    x = max_pool(x, pool_size=4)
    for i in range(4):
        x = double_conv_res_block(x, conv_pattern='[3, 96]')
    x = max_pool(x, pool_size=4)
    for i in range(6):
        x = double_conv_res_block(x, conv_pattern='[3, 192]')
    x = max_pool(x, pool_size=4)
    for i in range(3):
        x = double_conv_res_block(x, conv_pattern='[3, 384]')
    x = max_pool(x, pool_size=4)
    x = global_average_pooling(x, input_dim=384, output_dim=num_classes)
    x = Dense(num_classes)(x)
    return x


def m18(x, num_classes=10):
    x = conv(x, pattern='[80/4, 64]')
    x = max_pool(x, pool_size=4)
    x = conv(x, pattern='[3, 64] × 2')
    x = max_pool(x, pool_size=4)
    x = conv(x, pattern='[3, 128] × 4')
    x = max_pool(x, pool_size=4)
    x = conv(x, pattern='[3, 256] × 4')
    x = max_pool(x, pool_size=4)
    x = conv(x, pattern='[3, 512] × 4')
    x = global_average_pooling(x, input_dim=512, output_dim=num_classes)
    x = Dense(num_classes)(x)
    return x


def m11(x, num_classes=10):
    x = conv(x, pattern='[80/4, 64]')
    x = max_pool(x, pool_size=4)
    x = conv(x, pattern='[3, 64] × 2')
    x = max_pool(x, pool_size=4)
    x = conv(x, pattern='[3, 128] × 2')
    x = max_pool(x, pool_size=4)
    x = conv(x, pattern='[3, 256] × 2')
    x = max_pool(x, pool_size=4)
    x = conv(x, pattern='[3, 512] × 2')
    x = global_average_pooling(x, input_dim=512, output_dim=num_classes)
    x = Dense(num_classes)(x)
    return x


def m5_big(x, num_classes=10):
    x = conv(x, pattern='[80/4, 256]')
    x = max_pool(x, pool_size=4)
    x = conv(x, pattern='[3, 256]')
    x = max_pool(x, pool_size=4)
    x = conv(x, pattern='[3, 512]')
    x = max_pool(x, pool_size=4)
    x = conv(x, pattern='[3, 1024]')
    x = max_pool(x, pool_size=4)
    x = global_average_pooling(x, input_dim=1024, output_dim=num_classes)
    x = Dense(num_classes)(x)
    return x


def m5(x, num_classes=10):
    x = conv(x, pattern='[80/4, 128]')
    x = max_pool(x, pool_size=4)
    x = conv(x, pattern='[3, 128]')
    x = max_pool(x, pool_size=4)
    x = conv(x, pattern='[3, 256]')
    x = max_pool(x, pool_size=4)
    x = conv(x, pattern='[3, 512]')
    x = max_pool(x, pool_size=4)
    x = global_average_pooling(x, input_dim=512, output_dim=num_classes)
    x = Dense(num_classes)(x)
    return x


def m3_big(x, num_classes=10):
    x = conv(x, pattern='[80/4, 384]')
    x = max_pool(x, pool_size=4)
    x = conv(x, pattern='[3, 384]')
    x = max_pool(x, pool_size=4)
    x = global_average_pooling(x, input_dim=384, output_dim=num_classes)
    x = Dense(num_classes)(x)
    return x


def m3(x, num_classes=10):
    x = conv(x, pattern='[80/4, 256]')
    x = max_pool(x, pool_size=4)
    x = conv(x, pattern='[3, 256]')
    x = max_pool(x, pool_size=4)
    x = global_average_pooling(x, 256, num_classes)
    x = Dense(num_classes)(x)
    return x


def example(x, num_classes=2, keep_prob=0.5):
    x = Conv2D(92, kernel_size=(11, 11), strides=(4, 4), padding='same')(x)  # conv 1
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # LRN is missing here - Caffe.
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)  # pool 1

    x = Conv2D(256, kernel_size=(5, 5), padding='same')(x)  # miss group and pad param # conv 2
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)  # pool 2

    x = Conv2D(384, kernel_size=(3, 3), padding='same')(x)  # conv 3
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = MaxPooling2D(pool_size=(3, 3))(x)

    x = Conv2D(384, kernel_size=(3, 3), padding='same')(x)  # conv 4
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = MaxPooling2D(pool_size=(3, 3))(x)

    x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)  # conv 5
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Flatten()(x)
    x = Dense(4096, kernel_initializer='normal')(x)  # fc6
    # dropout 0.5
    x = tf.nn.dropout(x, keep_prob=keep_prob)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(4096, kernel_initializer='normal')(x)  # fc7
    # dropout 0.5
    x = tf.nn.dropout(x, keep_prob=keep_prob)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(num_classes)(x)
    # x = BatchNormalization()(x)
    # x = Activation('softmax')(x)
    return x

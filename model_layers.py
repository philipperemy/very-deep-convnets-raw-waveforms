import tensorflow as tf
from keras import regularizers
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.core import Activation, Flatten, Dense
from keras.layers.normalization import BatchNormalization

from constants import print_delimiter


def global_average_pooling(x, input_dim, output_dim):
    gap = tf.reduce_mean(x, 1)
    print_delimiter()
    print('GAP - input_dim = {}, output_dim = {}'.format(input_dim, output_dim))
    with tf.variable_scope('GAP'):
        gap_w = tf.get_variable('W', shape=[input_dim, output_dim], initializer=tf.random_normal_initializer(0., 0.01))
    logits = tf.matmul(gap, gap_w)
    return logits


# [receptive_field/stride, nb_filters] x 2
def conv_pattern_extractor(pattern):
    pattern = pattern.replace('x', '*')
    pattern = pattern.strip()
    num_blocks = 1
    if '*' in pattern:
        pattern, p2 = pattern.split('*')
        num_blocks = int(p2)
    pattern = pattern.strip()[1:-1]
    p1, p2 = pattern.split(',')
    nb_filters = int(p2)
    if '/' in p1:
        receptive_field, strides = [int(v) for v in p1.split('/')]
    else:
        receptive_field = int(p1)
        strides = 1
    return nb_filters, receptive_field, strides, num_blocks


def conv(x, pattern, batch_norm=True, relu=True):
    nb_filters, receptive_field, strides, num_blocks = conv_pattern_extractor(pattern)
    return conv_(x, nb_filters, receptive_field, strides, num_blocks, batch_norm, relu)


def conv_(x, nb_filters=128, receptive_field=3, strides=1, num_blocks=1, batch_norm=True, relu=True):
    for i in range(num_blocks):
        x = Conv1D(nb_filters,
                   kernel_size=receptive_field,
                   strides=strides,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.0001))(x)
        print_delimiter()
        print('Conv1D - filters = {}, kernel_size = {}, strides = {}'.format(nb_filters, receptive_field, strides))
        if batch_norm:
            x = BatchNormalization()(x)
            print_delimiter()
            print('Batch Normalization')
        if relu:
            x = Activation('relu')(x)
            print_delimiter()
            print('ReLU')
    return x


def max_pool(x, pool_size=4, strides=None):
    print_delimiter()
    print('MaxPooling1D - pool_size = {}, strides = {}'.format(pool_size, strides))
    x = MaxPooling1D(pool_size=pool_size, strides=strides)(x)
    return x


def double_conv_res_block(x, conv_pattern):
    # Fig. 1.
    print_delimiter()
    print('ResidualBlock')
    res_input = x
    x = conv(x, conv_pattern)
    x = conv(x, conv_pattern, relu=False)
    x += res_input
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


# to replace Global Average Pooling (GAP)
# Dropout 0.3
# TODO: check the order between BatchNormalization, Dropout.
# TODO: check any ReLU?
def fc_block(x, keep_prob=0.7):
    output_units = 1000
    print_delimiter()
    print('FC Block - {0} Dropout BN ReLU {0} Dropout BN ReLU'.format(output_units))
    x = Flatten()(x)
    x = Dense(output_units, kernel_initializer='glorot_uniform')(x)
    x = tf.nn.dropout(x, keep_prob=keep_prob)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(output_units, kernel_initializer='glorot_uniform')(x)
    x = tf.nn.dropout(x, keep_prob=keep_prob)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

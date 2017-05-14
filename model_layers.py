import tensorflow as tf
from keras import regularizers
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.core import Activation, Flatten, Dense
from keras.layers.normalization import BatchNormalization


def global_average_pooling(x, input_dim, output_dim):
    gap = tf.reduce_mean(x, 1)
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
        if batch_norm:
            x = BatchNormalization()(x)
        if relu:
            x = Activation('relu')(x)
    return x


def max_pool(x, pool_size=4, strides=None):
    x = MaxPooling1D(pool_size=pool_size, strides=strides)(x)
    return x


def double_conv_res_block(x, conv_pattern):
    # Fig. 1.
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
    x = Flatten()(x)
    x = Dense(1000, kernel_initializer='glorot_uniform')(x)
    x = tf.nn.dropout(x, keep_prob=keep_prob)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(1000, kernel_initializer='glorot_uniform')(x)
    x = tf.nn.dropout(x, keep_prob=keep_prob)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

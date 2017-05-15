import keras.backend as K

from constants import *
from model_data import DataReader
from models import *

if __name__ == '__main__':

    data_reader = DataReader()
    batch_size = 32  # TODO: I contacted the authors. what value do they use?
    num_classes = 10
    x = tf.placeholder(tf.float32, shape=[None, AUDIO_LENGTH, 1])
    y = tf.placeholder(tf.int64, [None])
    keep_prob = tf.placeholder(tf.float32)

    logits = m34_res(x, num_classes=num_classes)
    print_delimiter()
    print_total_trainable_parameters_count()

    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(logits, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(int(1e9)):
        batch_xs, batch_ys = data_reader.next_batch_train(batch_size)
        tr_loss, tr_acc, _ = sess.run([cross_entropy, accuracy, train_step],
                                      feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7,
                                                 K.learning_phase(): True})
        print('[TRAINING] #tr_batch = {0}, tr_loss = {1:.3f}, tr_acc = {2:.3f}'.format(i, tr_loss, tr_acc))
        if i % 100 == 0:
            batch_xt, batch_yt = data_reader.next_batch_test(batch_size)
            te_loss, te_acc = sess.run([cross_entropy, accuracy],
                                       feed_dict={x: batch_xt, y: batch_yt, keep_prob: 1.0,
                                                  K.learning_phase(): False})
            print('[TESTING] #te_batch = {0}, te_loss = {1:.3f}, te_acc = {2:.3f}'.format(i, te_loss, te_acc))

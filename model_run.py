from collections import deque

import numpy as np
from keras.utils.np_utils import to_categorical

from file_logger import FileLogger
from model_data import DataReader
from models import *


def deq():
    return deque(maxlen=100)


if __name__ == '__main__':

    file_logger = FileLogger('out.tsv', ['step', 'tr_loss', 'te_loss',
                                         'tr_acc', 'te_acc'])
    tr_loss_list, te_loss_list, tr_acc_list, te_acc_list = deq(), deq(), deq(), deq()
    data_reader = DataReader()
    batch_size = 128
    num_classes = 10

    model = m3(num_classes=num_classes)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())

    data_reader.train_files_count() * 400 / batch_size

    for i in range(int(1e9)):
        batch_xs, batch_ys = data_reader.next_batch_train(batch_size)
        batch_ys = to_categorical(batch_ys, num_classes=num_classes)
        tr_loss, tr_acc = model.train_on_batch(x=batch_xs, y=batch_ys)
        tr_loss_list.append(tr_loss)
        tr_acc_list.append(tr_acc)
        print('[TRAINING] #tr_batch = {0}, tr_loss = {1:.3f}, tr_acc = {2:.3f}'.format(i, tr_loss, tr_acc))

        batch_xt, batch_yt = data_reader.next_batch_test(batch_size)
        batch_yt = to_categorical(batch_yt, num_classes=num_classes)
        te_loss, te_acc = model.test_on_batch(x=batch_xt, y=batch_yt)
        te_loss_list.append(te_loss)
        te_acc_list.append(te_acc)
        print('[TESTING] #te_batch = {0}, te_loss = {1:.3f}, te_acc = {2:.3f}'.format(i, te_loss, te_acc))
        file_logger.write([i,
                           np.mean(tr_loss_list),
                           np.mean(te_loss_list),
                           np.mean(tr_acc_list),
                           np.mean(te_acc_list)])
    file_logger.close()

import sys

from keras.callbacks import Callback
from keras.callbacks import ReduceLROnPlateau
from keras.utils.np_utils import to_categorical

from file_logger import FileLogger
from model_data import DataReader
from model_resnet import resnet_34
from models import *


class MetricsHistory(Callback):
    def on_epoch_end(self, epoch, logs={}):
        file_logger.write([str(epoch),
                           str(logs['loss']),
                           str(logs['val_loss']),
                           str(logs['acc']),
                           str(logs['val_acc'])])


if __name__ == '__main__':
    model_name = 'm3'
    args = sys.argv
    if len(args) == 2:
        model_name = args[1].lower()
    print('Model selected:', model_name)
    file_logger = FileLogger('out_{}.tsv'.format(model_name), ['step', 'tr_loss', 'te_loss',
                                                               'tr_acc', 'te_acc'])
    model = None
    num_classes = 10
    if model_name == 'm3':
        model = m3(num_classes=num_classes)
    elif model_name == 'm5':
        model = m5(num_classes=num_classes)
    elif model_name == 'm11':
        model = m11(num_classes=num_classes)
    elif model_name == 'm18':
        model = m18(num_classes=num_classes)
    elif model_name == 'm34':
        model = resnet_34(num_classes=num_classes)

    if model is None:
        exit('Please choose a valid model: [m3, m5, m11, m18, m34]')

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())

    data_reader = DataReader()
    x_tr, y_tr = data_reader.get_all_training_data()
    y_tr = to_categorical(y_tr, num_classes=num_classes)
    x_te, y_te = data_reader.get_all_testing_data()
    y_te = to_categorical(y_te, num_classes=num_classes)

    print('x_tr.shape =', x_tr.shape)
    print('y_tr.shape =', y_tr.shape)
    print('x_te.shape =', x_te.shape)
    print('y_te.shape =', y_te.shape)

    # if the accuracy does not increase over 10 epochs, we reduce the learning rate by half.
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, min_lr=0.0001, verbose=1)
    metrics_history = MetricsHistory()
    batch_size = 128
    model.fit(x=x_tr,
              y=y_tr,
              batch_size=batch_size,
              epochs=400,
              verbose=1,
              shuffle=True,
              validation_data=(x_te, y_te),
              callbacks=[metrics_history, reduce_lr])

    file_logger.close()

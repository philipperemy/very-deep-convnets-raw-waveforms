import pickle
from glob import glob
from random import choice
from time import time

import librosa
import numpy as np

from constants import *


class DataReader:
    def __init__(self):
        self.train_files = glob(os.path.join(OUTPUT_DIR_TRAIN, '**.pkl'))
        print('training files =', len(self.train_files))
        self.test_files = glob(os.path.join(OUTPUT_DIR_TEST, '**.pkl'))
        print('testing files =', len(self.test_files))

    def next_batch_train(self, batch_size):
        return DataReader._next_batch(batch_size, self.train_files)

    def next_batch_test(self, batch_size):
        return DataReader._next_batch(batch_size, self.test_files)

    @staticmethod
    def _next_batch(batch_size, file_list):
        x, y = [], []
        for i in range(batch_size):
            filename = choice(file_list)
            with open(filename, 'rb') as f:
                audio_element = pickle.load(f)
                x.append(audio_element['audio'])
                y.append(int(audio_element['class_id']))
        return np.array(x), np.array(y)


def read_audio_from_filename(filename, target_sr):
    audio, _ = librosa.load(filename, sr=target_sr, mono=True)
    audio = audio.reshape(-1, 1)
    return audio


def next_batch_blank(batch_size):
    return np.zeros(shape=(batch_size, AUDIO_LENGTH, 1), dtype=np.float32), np.ones(shape=batch_size)


if __name__ == '__main__':
    read_audio_from_filename('samples/15564-2-0-0.wav', target_sr=TARGET_SR)
    data_reader = DataReader()
    a = time()
    data_reader.next_batch_train(128)
    print(time() - a, 'sec')
    data_reader.next_batch_test(32)

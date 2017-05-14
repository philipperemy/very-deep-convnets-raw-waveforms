import os

DATA_AUDIO_DIR = '/Users/philipperemy/Downloads/UrbanSound8K/audio'
TARGET_SR = 8000
OUTPUT_DIR = '/tmp/very-deep-conv-nets-raw-waveforms'
OUTPUT_DIR_TRAIN = os.path.join(OUTPUT_DIR, 'train')
OUTPUT_DIR_TEST = os.path.join(OUTPUT_DIR, 'test')

AUDIO_LENGTH = 32000
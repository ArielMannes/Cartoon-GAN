from builtins import filter
from glob import glob
from keras.models import Model
from keras.layers import *


class CartoonGAN(object):
    def __init__(self):
        self.model_name = 'CartoonGAN'

        self.trainA_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainA'))
        self.trainB_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainB'))
        self.trainB_smooth_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainB_smooth'))

        self.dataset_num = max(len(self.trainA_dataset), len(self.trainB_dataset))

    def generator(self):
        inp = Input(shape=(None, None, 3), name='gen_INP')
        ##PRE
        pre = Conv2D(kernel_size=7, filters=64,strides=1 )

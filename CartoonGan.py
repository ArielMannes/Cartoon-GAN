from builtins import filter
from glob import glob
from keras.models import Model
from keras.layers import *
from keras_contrib.layers.normalization import instancenormalization
import tensorflow as tf


class CartoonGAN(object):
    def __init__(self, dasetname, num_filters=64):
        self.num_filters = num_filters
        self.model_name = 'CartoonGAN'
        self.dataset_name = dasetname

        self.trainA_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainA'))
        self.trainB_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainB'))
        self.trainB_smooth_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainB_smooth'))

        self.dataset_num = max(len(self.trainA_dataset), len(self.trainB_dataset))

    def res_block(self):
        inp = Input(shape=(None, None, self.num_filters * 4))
        x = Conv2D(kernel_size=3, filters=self.num_filters * 4, strides=1)(inp)
        x = instancenormalization.InstanceNormalization()(x)
        x = ReLU()(x)

        x = Conv2D(kernel_size=3, filters=self.num_filters * 4, strides=1)(x)
        x = instancenormalization.InstanceNormalization()(x)
        return Model(inp, add([inp, x]))

    def generator(self):
        gen_filters = self.num_filters
        inp = Input(shape=(None, None, 3), name='gen_INP')
        # PRE
        x = Conv2D(kernel_size=7, filters=gen_filters, strides=1, padding='same')(inp)
        x = instancenormalization.InstanceNormalization()(x)
        x = ReLU()(x)

        # Down-Sampling
        # pt1
        x = Conv2D(kernel_size=3, filters=gen_filters * 2, strides=2)(x)
        x = Conv2D(kernel_size=3, filters=gen_filters * 2, strides=1)(x)
        x = instancenormalization.InstanceNormalization()(x)
        x = ReLU()(x)

        # pt2
        x = Conv2D(kernel_size=3, filters=gen_filters * 4, strides=2)(x)
        x = Conv2D(kernel_size=3, filters=gen_filters * 4, strides=1)(x)
        x = instancenormalization.InstanceNormalization()(x)
        x = ReLU()(x)

        # residual blocks

        for i in range(8):
            x = self.res_block()(x)

        # up convolution
        x = Conv2DTranspose(filters=self.num_filters * 2, kernel_size=3, strides=2, )(x)
        x = Conv2D(kernel_size=3, filters=self.num_filters * 2, strides=1, )(x)
        x = instancenormalization.InstanceNormalization()(x)
        x = ReLU()(x)

        x = Conv2DTranspose(filters=self.num_filters, kernel_size=3, strides=2, )(x)
        x = Conv2D(kernel_size=3, filters=self.num_filters, strides=1)(x)
        x = instancenormalization.InstanceNormalization()(x)
        x = ReLU()(x)

        x = Conv2D(kernel_size=7, filters=3, strides=1, activation='tanh')(x)

        return Model(inp, x)

    def discriminator(self):
        inp = Input(shape=(None, None, 3))
        x = Conv2D(kernel_size=3, filters=self.num_filters // 2, strides=1, use_bias=False)(inp)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(kernel_size=3, filters=self.num_filters, strides=2, use_bias=False)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(kernel_size=3, filters=self.num_filters * 2, strides=1, use_bias=False)(x)
        x = instancenormalization.InstanceNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(kernel_size=3, filters=self.num_filters * 2, strides=2, use_bias=False)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(kernel_size=3, filters=self.num_filters * 4, strides=1, use_bias=False)(x)
        x = instancenormalization.InstanceNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(kernel_size=3, filters=self.num_filters * 4, strides=1, use_bias=False)(x)
        x = instancenormalization.InstanceNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(kernel_size=3, filters=1, strides=1, use_bias=False)(x)

        return Model(inp, x)

    def sigmoid_cross_entropy_with_logistics(self,x , like_func):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=like_func(x),logits=x))


    def discriminator(self,*factors):
        return sum(map(self.sigmoid_cross_entropy_with_logistics, factors, [tf.ones_like, tf.zeros_like, tf.zeros_like]))


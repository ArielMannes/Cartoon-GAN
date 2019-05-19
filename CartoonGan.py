from glob import glob
import Utils.vgg_loss_model as loss_model
from keras.models import Model
from keras.layers import *
from keras_contrib.layers.normalization import instancenormalization
from math import ceil

from Utils.Loss import generator_loss, vgg_loss, discriminator_loss, smooth_gen


class CartoonGAN(object):
    def __init__(self, dasetname, num_filters=64):
        self.num_filters = num_filters
        self.model_name = 'CartoonGAN'
        self.dataset_name = dasetname

        self.train_flicker_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainA'))
        self.train_cartoon_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainB'))
        self.train_cartoon_smooth_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainB_smooth'))

        self.dataset_num = max(len(self.train_flicker_dataset), len(self.train_cartoon_dataset))
        self.conv4_4 = loss_model.get_conv_4_4

        self.Generator = self.generator()
        self.Discriminator = self.discriminator()
        self.data_path = ''

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

    def pre_train(self, num_epochs=10):
        x = np.load('{}/Numpy_arrays/flicker_train.npy'.format(self.data_path))
        self.Generator.compile(optimizer='adam', loss=vgg_loss(self.conv4_4))
        self.Generator.fit(x=x, y=x, epochs=num_epochs)

    def train_model(self, epochs=40, batch_size=32):
        flicker_train = np.load('{}/Numpy_arrays/flicker_train.npy'.format(self.data_path))
        cartoon_train = np.load('{}/Numpy_arrays/spirited.npy'.format(self.data_path))
        cartoon_smooth = np.load('{}/Numpy_arrays/spirited_smooth.npy'.format(self.data_path))

        cartoon_smooth_gen = smooth_gen(cartoon_smooth, batch_size)
        self.Generator.compile(optimizer='adam', loss=generator_loss(self.conv4_4, self.Discriminator))
        self.Discriminator.compile(optimizer='adam', loss=discriminator_loss(cartoon_smooth_gen))

        for epoch in range(epochs):
            for batch in range(ceil(len(cartoon_train) / batch_size)):
                p_batch = flicker_train[batch * batch_size: (batch + 1) * batch_size]
                c_batch = cartoon_train[batch * batch_size: (batch + 1) * batch_size]

                g_p = self.Generator.predict(p_batch)
                x = np.concatenate((c_batch, g_p))
                y = np.ones([2 * batch_size, 1])
                y[batch_size:, :] = 0
                self.Discriminator.train_on_batch(x, y)
                self.Generator.train_on_batch(p_batch, p_batch)

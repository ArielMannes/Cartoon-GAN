from math import ceil

import tensorflow as tf


def sigmoid_cross_entropy_with_logits(x, like_func):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=like_func(x), logits=x))


def discriminator_loss(real_blur, adv_weight=10):
    def loss(real, fake):
        return adv_weight * sum(map(sigmoid_cross_entropy_with_logits,
                                    [real, fake, real_blur.__next__()],
                                    [tf.ones_like, tf.zeros_like, tf.zeros_like]))

    return loss


def g_loss(fake):
    return sigmoid_cross_entropy_with_logits(fake, tf.ones_like)


def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss


def vgg_loss(real, fake):
    return L1_loss(real, fake)


def generator_loss(conv4_4, disc, adv_weight=10):
    def loss(real, fake):
        return vgg_loss(conv4_4)(real, fake) + (adv_weight * g_loss(disc.predict(fake)))

    return loss


def smooth_gen(cartoon_smooth, batch_size):
    while True:
        for batch in range(int(ceil(len(cartoon_smooth) / batch_size))):
            yield cartoon_smooth[batch * batch_size: (batch + 1) * batch_size]

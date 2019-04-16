import tensorflow as tf


def sigmoid_cross_entropy_with_logits(x, like_func):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=like_func(x), logits=x))


def discriminator_loss(real, fake, real_blur):
    return sum(map(sigmoid_cross_entropy_with_logits,
                   [real, fake, real_blur],
                   [tf.ones_like, tf.zeros_like, tf.zeros_like]))


def generator_loss(fake):
    return sigmoid_cross_entropy_with_logits(fake, tf.ones_like)


def vgg_loss(real, fake):
    # TODO: implement
    return NotImplementedError

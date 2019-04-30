import keras
import tensorflow as tf



def sigmoid_cross_entropy_with_logits(x, like_func):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=like_func(x), logits=x))


def discriminator_loss(real, fake, real_blur):
    return sum(map(sigmoid_cross_entropy_with_logits,
                   [real, fake, real_blur],
                   [tf.ones_like, tf.zeros_like, tf.zeros_like]))


def generator_loss(fake):
    return sigmoid_cross_entropy_with_logits(fake, tf.ones_like)

def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss



def vgg_loss(real, fake, conv4_4):

    real_features = conv4_4.evaluate(real)
    fake_features = conv4_4.evaluate(fake)

    return L1_loss(real_features, fake_features)



vgg_loss(None, None)
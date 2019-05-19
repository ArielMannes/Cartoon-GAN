import keras
from keras.models import Model


def get_conv_4_4():
    vgg19 = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None,
                                           input_shape=(225, 225, 3), pooling=None, classes=1000)
    for l in vgg19.layers:
        l.trainable = False
    conv4_4 = Model(vgg19.input, vgg19.layers[15].output)
    return conv4_4
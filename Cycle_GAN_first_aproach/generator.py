from random import random
from numpy import load
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization 
from matplotlib import pyplot


def resnet_block(n_filters, input_layer):

 init = RandomNormal(stddev=0.02)

 g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
 g = InstanceNormalization(axis=-1)(g)
 g = Activation('relu')(g)
 g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
 g = InstanceNormalization(axis=-1)(g)
 g = Concatenate()([g, input_layer])
 return g
 

def define_generator(image_shape, n_resnet=9):
 init = RandomNormal(stddev=0.02)

 in_image = Input(shape=image_shape)


 g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
 g = InstanceNormalization(axis=-1)(g)
 g = Activation('relu')(g)
 
 g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
 g = InstanceNormalization(axis=-1)(g)
 g = Activation('relu')(g)
 
 g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
 g = InstanceNormalization(axis=-1)(g)
 g = Activation('relu')(g)

 for _ in range(n_resnet):
    g = resnet_block(256, g)

 g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
 g = InstanceNormalization(axis=-1)(g)
 g = Activation('relu')(g)
 g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
 g = InstanceNormalization(axis=-1)(g)
 g = Activation('relu')(g)
 g = Conv2D(3, (7,7), padding='same', kernel_initializer=init)(g)
 g = InstanceNormalization(axis=-1)(g)
 out_image = Activation('tanh')(g)
 model = Model(in_image, out_image)
 return model


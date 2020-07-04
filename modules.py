import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

def build_generator():
    generator = Sequential([
        # 100 dims input noise > 7*7*256 output
        Dense(7*7*256, use_bias=False, input_shape=(100,)),
        BatchNormalization(),
        LeakyReLU(),

        # Reshape (7, 7, 256)
        Reshape((7, 7, 256)),

        # Upsampling
        Conv2DTranspose(128, (5,5), strides=(1,1), padding="same", use_bias=False),
        BatchNormalization(),
        LeakyReLU(),

        # Upsampling
        Conv2DTranspose(64, (5,5), strides=(2,2), padding="same", use_bias=False),
        BatchNormalization(),
        LeakyReLU(),

        # to grayscale
        Conv2DTranspose(1, (5,5), strides=(2,2), padding="same", use_bias=False)
    ])
    return generator

def build_discriminator():
    discriminator = Sequential([
        Conv2D(64, (5,5), strides=(2,2), padding="same", input_shape=(28,28,1)),
        LeakyReLU(),
        Dropout(.3),

        Conv2D(128, (5,5), strides=(2,2), padding="same"),
        LeakyReLU(),
        Dropout(.3),

        Flatten(),
        Dense(1)
    ])
    return discriminator

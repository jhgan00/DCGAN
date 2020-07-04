import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

# Generator
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

# Discriminator
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

# Optimizers
generator_optimizer = Adam(1e-4)
discriminator_optimizer = Adam(1e-4)


# Losses
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator
)

checkpoint_manager = tf.train.CheckpointManager(checkpoint, "./training_checkpoints")
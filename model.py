import modules as md
import tensorflow as tf
import os
from time import time

class DCGAN:
    def __init__(self, batch_size=256, noise_dim=100):
        self.generator = md.generator
        self.gen_optimizer = md.generator_optimizer
        self.discriminator = md.discriminator
        self.disc_optimizer = md.discriminator_optimizer
        self.checkpoint_dir = "./training_checkpoints"
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = md.checkpoint
        self.batch_size = batch_size
        self.noise_dim = noise_dim

    def train_step(self, images):
        noise = tf.random.normal([self.batch_size, self.noise_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_images = self.generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = md.generator_loss(fake_output)
            disc_loss = md.discriminator(real_output, fake_output)

        grads_gen = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        grads_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(grads_gen, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(grads_disc, self.generator.trainable_variables))

    def train(self, dataset, epochs):
        for epoch in range(epochs):
            start = time.time()
            for batch in dataset:
                self.train_step(batch)
            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix=checkpoint_prefix)
            print(f"Time for epoch {epoch} is {time()-start} sec")
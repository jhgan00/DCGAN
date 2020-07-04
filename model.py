import dataset
import modules as md
import tensorflow as tf
import os
from time import time
from tqdm import tqdm
import datetime
import io
import matplotlib.pyplot as plt
import numpy as np

class DCGAN:
    def __init__(self, batch_size=256, noise_dim=100):
        self.generator = md.generator
        self.gen_optimizer = md.generator_optimizer
        self.discriminator = md.discriminator
        self.disc_optimizer = md.discriminator_optimizer
        self.checkpoint_dir = "./training_checkpoints"
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = md.checkpoint
        self.batch_size = batch_size
        self.steps_per_epoch = int(60000/batch_size)
        self.noise_dim = noise_dim
        self.dataset = dataset.create_dataset(self.batch_size)

    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image


    def image_grid(self, generated_images):
        """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
        # Create a figure to contain the plot.
        images = np.squeeze(generated_images.numpy())
        figure = plt.figure(figsize=(5, 1))
        for i in range(5):
            # Start next subplot.
            plt.subplot(1, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images[i], cmap=plt.cm.binary)
        return figure

    def train_step(self, images):
        noise = tf.random.normal([self.batch_size, self.noise_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = md.generator_loss(fake_output)
            disc_loss = md.discriminator_loss(real_output, fake_output)

        grads_gen = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        grads_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(grads_gen, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(grads_disc, self.discriminator.trainable_variables))

        return round(gen_loss.numpy(),3), round(disc_loss.numpy(), 3)

    def train(self, epochs):
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
        for epoch in range(epochs):
            start = time()
            with tqdm(total=self.steps_per_epoch) as progress_bar:
                for batch in self.dataset:
                    gen_loss, disc_loss = self.train_step(batch)
                    progress_bar.update(1)  # update progress
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            tqdm.write(f"Epoch: {epoch+1}   Time: {round(time()-start)}sec  G: {round(gen_loss,3)} D: {round(disc_loss,3)}")
            test_input = tf.random.normal([5, self.noise_dim])
            img = self.generator(test_input, training=False)
            fig = self.plot_to_image(self.image_grid(img))

            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

            train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)

            with train_summary_writer.as_default():
                tf.summary.scalar('Generator loss', gen_loss, step=epoch)
                tf.summary.scalar('Discriminator loss', disc_loss, step=epoch)
                tf.summary.image("Generated Image", fig, step=epoch)
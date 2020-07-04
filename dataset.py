import tensorflow as tf

def create_dataset(batch_size, buffer_size=60000):
    print("loading dataset")
    (train_images, test_images), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape((-1, 28, 28, 1)).astype("float32")
    train_images = (train_images-127.5)/127.5
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batch_size)
    print("successfully loaded dataset")
    return train_dataset
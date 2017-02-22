import tensorflow as tf

# Load MNIST data
from tensorflow.example.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Convolution layer 1
conv1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=32,
    kernel_size=5,
    padding="same",
    activation=tf.nn.relu)

# Pool layer 1
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size = [2, 2], strides = 2)

# Convolution layer 2
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=5,
    padding="same",
    activation=tf.nn.relu)

# Pool layer 2
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size = [2, 2], strides = 2)

# Dense layer


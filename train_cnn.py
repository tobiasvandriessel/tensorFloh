# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


# Our application logic will be added here

if __name__ == "__main__":
  tf.app.run()


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 120, 120, 3])

    conv1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=96,
    kernel_size=[7, 7],
    padding="same",
    activation=tf.nn.relu,
    strides=(2,2))

    norm1 = tf.nn.lrn(conv1, 5, 2, 0.0001, 0.75)

    pool1 = tf.layers.MaxPooling2D(inputs=norm1, pool_size=[2,2], strides=2)


    conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=256,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu,
    strides=(2,2))

    norm2 = tf.nn.lrn(conv2, 5, 2, 0.0001, 0.75)

    pool2 = tf.layers.MaxPooling2D(inputs=norm2, pool_size=[2,2], strides=2)


    conv3 = tf.layers.conv2d(
    inputs=pool2,
    filters=512,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu)


    conv4 = tf.layers.conv2d(
    inputs=conv3,
    filters=512,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu)

    conv5 = tf.layers.conv2d(
    inputs=pool2,
    filters=512,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu)


    pool3 = tf.layers.MaxPooling2D(inputs=conv5, pool_size=[2,2], strides=2)

    pool3_flat = tf.reshape(pool3, [-1, 15 * 15 * 512])

    #Not sure if relu ois the correct activation function, probably in paper
    dense1 = tf.layers.dense(inputs=pool3_flat, units=4096, activation=tf.nn.relu)
    
    #Random rate now, in paper is correct one
    dropout1 = tf.layers.dropout(inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    
    #Not sure if relu ois the correct activation function, probably in paper
    dense2 = tf.layers.dense(inputs=dropout1, units=2048, activation=tf.nn.relu)
    
    #Random rate now, in paper is correct one
    dropout2 = tf.layers.dropout(inputs=dense2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout2, units=5)

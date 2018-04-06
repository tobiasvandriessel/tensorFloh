# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


# Our application logic will be added here

if __name__ == "__main__":
  tf.app.run()


# About learning rate: The learning rate is initially set to 10e−2, and then decreased according to a fixed schedule, 
#  which is kept the same for all training sets. Namely, when training a ConvNet from scratch, the rate is changed to 10e−3 
#  after 50K iterations, then to 10e−4 after 70K iterations, and training is stopped after 80K iterations. 
#  In the fine-tuning scenario, the rate is changed to 10e−3 after 14K iterations, and training stopped after 20K iterations.

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 120, 120, 3])


    #I don't think our first layer needs this big kernel size, probably too big for our input size. Maybe 5 is enough
    conv1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=96,
    kernel_size=[7, 7],
    padding="same",
    activation=tf.nn.relu,
    strides=(2,2))

    norm1 = tf.nn.lrn(conv1, 5, 2, 0.0001, 0.75)

    pool1 = tf.layers.MaxPooling2D(inputs=norm1, pool_size=[3,3], strides=2)


    conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=256,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu,
    strides=(2,2))

    norm2 = tf.nn.lrn(conv2, 5, 2, 0.0001, 0.75)

    pool2 = tf.layers.MaxPooling2D(inputs=norm2, pool_size=[3,3], strides=2)


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


    pool3 = tf.layers.MaxPooling2D(inputs=conv5, pool_size=[3,3], strides=2)

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

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
    activation=tf.nn.relu)

    norm1 = tf.nn.lrn(input_layer, 5, 2, 0.0001, 0.75)

    pool1 = tf.layers.MaxPooling2D(inputs=norm1, pool_size=[2,2], strides=2)

    



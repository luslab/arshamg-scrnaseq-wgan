import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class ScaleCell(layers.Layer):

    def __init__(self, scale_factor):
        super(ScaleCell, self).__init__()
        self.scale_factor = scale_factor

    def build(self, input_shape):
        self.gammas = tf.Variable(np.ones(input_shape[0],) * self.scale_factor, trainable=False, dtype=tf.float32)

    def call(self, inputs):
        # Calc the per cell sums
        sigmas = tf.reduce_sum(inputs, axis=1)

        # Calc scale val
        scale_val = self.gammas / (sigmas + sys.float_info.epsilon)

        # Scale
        return tf.transpose(tf.transpose(inputs) * scale_val)
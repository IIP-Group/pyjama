"""Projection onto Orthogonal Space (orthonal to jammer subspace)"""

import tensorflow as tf
import numpy as np
import sionna

class OrthogonalSubspaceProjector(tf.keras.layers.Layer):
    def __init__(self, dtype=tf.complex64, **kwargs):
        super().__init__(trainable=False, dtype=dtype, **kwargs)
        
        
    def set_jammer(self, j):
        """
        j: ([batch_size, num_rx, num_rx_ant, num_jammer, num_jammer_ant, num_ofdm_symbols, fft_size], tf.complex)
            Jammer channel responses
        """
        jammer_shape = tf.shape(j)
        # rearange dimensions of j to [..., num_rx * num_rx_ant, num_jammer * num_jammer_ant]. As tf.reshape is very cheap, we can use it multiple timmes
        j = tf.transpose(j, [0, 5, 6, 1, 2, 3, 4])
        j = sionna.utils.flatten_dims(j, 2, 3)
        j = sionna.utils.flatten_last_dims(j, 2)

        self._proj = tf.eye(jammer_shape[1] * jammer_shape[2], dtype=self.dtype) - tf.matmul(j, sionna.utils.matrix_pinv(j))


    def call(self, inputs):
        """
        Project onto orthogonal subspace of jammer subspace, given the jammer channel responses j

        inputs: y
        y: ([batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.complex) (frequency response) or
           ([batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex) (signal)

            Input. Might be e.g. signal (with jammer interference), or channel frequency response.

        Output: y_proj: same shape as y
        """

        y = inputs
        input_shape = tf.shape(y)

        # we can handle both y input types like the same by inserting dimensions for num_tx and num_tx_ant
        y = sionna.utils.expand_to_rank(y, 7, axis=3)
        # rearange dimensions of y to [..., num_rx * num_rx_ant, num_tx * num_tx_ant]
        y = tf.transpose(y, [0, 5, 6, 1, 2, 3, 4])
        before_flatten_shape = tf.shape(y)
        y = sionna.utils.flatten_dims(y, 2, 3)
        y = sionna.utils.flatten_last_dims(y, 2)
        
        y_proj = tf.matmul(self._proj, y)
        # unflatten dimensions and transpose back
        y_proj = tf.reshape(y_proj, before_flatten_shape)
        y_proj = tf.transpose(y_proj, [0, 3, 4, 5, 6, 1, 2])
        # undo rank exppansion
        y_proj = tf.reshape(y_proj, input_shape)
        
        return y_proj
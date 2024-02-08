"""Projection onto Orthogonal Space (orthonal to jammer subspace)"""

import tensorflow as tf
import numpy as np
import sionna
from ..utils import reduce_matrix_rank

class OrthogonalSubspaceProjector(tf.keras.layers.Layer):
    """
    Mitigates jammer interference by projecting onto the orthogonal subspace of the jammer subspace.
    Jammer subspace has to be set before calling the layer. This can be done by either calling `set_jammer_frequency_response` or `set_jammer_covariance`.

    This layer should be called on the received signal and, if the unmapped channel frequency response is used, on it before equalization.

    Parameters
    ----------
    dimensionality : int
        Rank of the jammer subspace which is "subtracted".
        If None, the maximum dimensionality is assumed (i.e. the rank of the jammer covariance matrix).
    dtype : tf.Dtype
        Defines the datatype for internal calculations and the output dtype.
        Defaults to `tf.complex64`.

    Input
    -----
    y : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.complex (frequency response) or\
       ([batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex) (signal)

        Input. Might be e.g. signal (including jammer interference), or channel frequency response (to be mapped to subspace).

    Output
    ------
        y_proj : Same shape as ``y``, dtype
            y projected onto the orthogonal subspace.
    """

    def __init__(self, dimensionality=None, dtype=tf.complex64, **kwargs):
        self._dimensionality = dimensionality
        super().__init__(trainable=False, dtype=dtype, **kwargs)
        
    def set_jammer_frequency_response(self, j):
        """
        Given the jammer channel frequency response j,
        precompute the projection matrix which maps the signal onto the orthogonal subspace of the jammer subspace.

        Input
        -----
        j : [batch_size, num_rx, num_rx_ant, num_jammer, num_jammer_ant, num_ofdm_symbols, fft_size], tf.complex
            Jammer channel frequency response
        """
        jammer_shape = tf.shape(j)
        # rearange dimensions of j to [..., num_rx_ant, num_jammer * num_jammer_ant].
        j = tf.transpose(j, [0, 1, 5, 6, 2, 3, 4])
        j = sionna.utils.flatten_last_dims(j, 2)

        # [batch_size, num_rx, num_ofdm_symbols, fft_size, num_rx_ant, num_rx_ant]
        j_j_pinv = tf.matmul(j, sionna.utils.matrix_pinv(j))
        if self._dimensionality is not None:
            j_j_pinv = reduce_matrix_rank(j_j_pinv, self._dimensionality)
            
        self._proj = tf.eye(jammer_shape[2], dtype=self.dtype) - j_j_pinv

    def set_jammer_covariance(self, jammer_covariance):
        """
        Input
        -----
        jammer_covariance : [batch_size, num_rx, num_ofdm_symbols, fft_size, num_rx_ant, num_rx_ant], tf.complex
            Covariance matrix of jammer signal.

        .. deprecated:: 0.1.0
            Use `set_jammer_signals` instead.
        """
        num_rx_ant = tf.shape(jammer_covariance)[-1]
        # jammer_covariance = jammer_covariance / sionna.utils.expand_to_rank(tf.linalg.trace(jammer_covariance), jammer_covariance.shape.rank, axis=-1)
        # if self._dimensionality is not None:
        #     jammer_covariance = reduce_matrix_rank(jammer_covariance, self._dimensionality)
        # self._proj = tf.eye(num_rx_ant, dtype=self.dtype) - jammer_covariance
        # TODO when doing it this way, we should use the jammer signals directly (and hence limit the rank reduction to num_jammer_symbols)
        _, u, _ = tf.linalg.svd(jammer_covariance, compute_uv=True)
        if self._dimensionality is not None:
            u = u[..., :self._dimensionality]
        self._proj = tf.eye(num_rx_ant, dtype=self.dtype) - tf.matmul(u, u, adjoint_b=True)

    def set_jammer_signals(self, jammer_signals):
        """
        Input
        -----
        jammer_signals : [batch_size, num_rx, num_rx_ant, num_symbols, fft_size], tf.complex
            Received symbols containing only jammer interference.
        """
        # -> [..., num_rx_ant, num_symbols]
        num_rx_ant = tf.shape(jammer_signals)[2]
        jammer_signals = tf.transpose(jammer_signals, [0, 1, 4, 2, 3])
        _, u, _ = tf.linalg.svd(jammer_signals, compute_uv=True)
        if self._dimensionality is not None:
            u = u[..., :self._dimensionality]
        proj = tf.eye(num_rx_ant, dtype=self.dtype) - tf.matmul(u, u, adjoint_b=True)
        # add dimension for ofdm symbols
        # [batch_size, num_rx, 1, fft_size, num_rx_ant, num_rx_ant]
        self._proj = tf.expand_dims(proj, axis=2)


    def call(self, inputs):

        y = inputs
        input_shape = tf.shape(y)

        # we can handle both y input types like the same by inserting dimensions for num_tx and num_tx_ant
        y = sionna.utils.expand_to_rank(y, 7, axis=3)
        # rearange dimensions of y to [..., num_rx_ant, num_tx * num_tx_ant]
        # e.g. (batch_size, num_rx, num_ofdm_symbols, fft_size, num_rx_ant, num_tx * num_tx_ant)
        y = tf.transpose(y, [0, 1, 5, 6, 2, 3, 4])
        before_flatten_shape = tf.shape(y)
        y = sionna.utils.flatten_last_dims(y, 2)
        
        y_proj = tf.matmul(self._proj, y)
        # unflatten dimensions and transpose back
        y_proj = tf.reshape(y_proj, before_flatten_shape)
        # y_proj = tf.transpose(y_proj, [0, 3, 4, 5, 6, 1, 2])
        y_proj = tf.transpose(y_proj, [0, 1, 4, 5, 6, 2, 3])
        # undo rank exppansion
        y_proj = tf.reshape(y_proj, input_shape)
        
        return y_proj
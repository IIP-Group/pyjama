"""LMMSE treating Interference as Noise (IAN)"""

import numpy as np
import tensorflow as tf
import sionna
from sionna.ofdm import OFDMEqualizer
from sionna.mimo import lmmse_equalizer


class IanLMMSEEqualizer(OFDMEqualizer):
    """Usage: Set jammer_covariance before __call__"""
    def __init__(self,
                 resource_grid,
                 stream_management,
                 whiten_interference=True,
                 dtype=tf.complex64,
                 **kwargs):

        # per default, we set the jammer covariance to the zero (broadcastable), i.e. we assume no jammer
        self.jammer_covariance = 0.0j

        def equalizer(y, h, s):
            return lmmse_equalizer(y, h, s + self.jammer_covariance, whiten_interference)

        super().__init__(equalizer=equalizer,
                         resource_grid=resource_grid,
                         stream_management=stream_management,
                         dtype=dtype, **kwargs)

    def set_covariance_matrix_from_jammer_frequency_response(self, j, rho):
        """Set the jammer covariance matrix from the frequency response of the jammer channel.
        j: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.complex, frequency response of jammer channel
        rho: jammer power
        """
        # TODO: incorporate jammer power (variance)
        # to [batch_size, rx, num_sym, fft_size, rx_ant, tx*tx_ant]
        j_dt = tf.transpose(j, perm=[0, 1, 5, 6, 2, 3, 4])
        j_dt = sionna.utils.flatten_last_dims(j_dt, 2)
        j_dt = tf.cast(j_dt, dtype=self._dtype)
        
        # [batch_size, rx, num_sym, fft_size, rx_ant, rx_ant]
        self.jammer_covariance = tf.matmul(j_dt, j_dt, adjoint_b=True)
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

    def set_covariance_matrix_from_jammer_frequency_response(self, h_freq, jammer_variance):
        """Set the jammer covariance matrix from the frequency response of the jammer channel.
        h_freq: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.complex
        """
"""LMMSE treating Interference as Noise (IAN)"""

import numpy as np
import tensorflow as tf
import sionna
from sionna.ofdm import OFDMEqualizer
from sionna.mimo import lmmse_equalizer
from sionna.utils import flatten_last_dims, expand_to_rank
from ..utils import reduce_matrix_rank


# TODO should we remove jammer variance from estimation of no_eff? Or is it correct to always include it?
class IanLMMSEEqualizer(OFDMEqualizer):
    # """Usage: Set jammer_covariance before __call__"""
    """
    LMMSE Equalizer mitigating jammer interference by treating it as noise.
    To be used instead of `~sionna.ofdm.LMMSEEqualizer` to equalize and supress jammer at the same time.
    
    One of `set_jammer` or `set_jammer_covariance` must be called before calling this layer.

    Parameters
    ----------
    resource_grid : ResourceGrid
        Instance of :class:`~sionna.ofdm.ResourceGrid`

    stream_management : StreamManagement
        Instance of :class:`~sionna.mimo.StreamManagement`

    whiten_interference : bool
        If `True` (default), the interference is first whitened before equalization.
        In this case, an alternative expression for the receive filter is used which
        can be numerically more stable.

    dtype : tf.Dtype
        Datatype for internal calculations and the output dtype.
        Defaults to `tf.complex64`.


    Input
    -----
    (y, h_hat, err_var, no) :
        Tuple:

    y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex
        Received OFDM resource grid after cyclic prefix removal and FFT

    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex
        Channel estimates for all streams from all transmitters

    err_var : [Broadcastable to shape of ``h_hat``], tf.float
        Variance of the channel estimation error

    no : [batch_size, num_rx, num_rx_ant] (or only the first n dims), tf.float
        Variance of the AWGN

    Output
    ------
    x_hat : [batch_size, num_tx, num_streams, num_data_symbols], tf.complex
        Estimated symbols with jammer interference removed

    no_eff : [batch_size, num_tx, num_streams, num_data_symbols], tf.float
        Effective noise variance for each estimated symbol (including jammer interference)
    """

    def __init__(self,
                 resource_grid,
                 stream_management,
                 whiten_interference=True,
                 dtype=tf.complex64,
                 **kwargs):

        # per default, we set the jammer covariance to the zero (broadcastable), i.e. we assume no jammer
        self._jammer_covariance = 0.0j

        def equalizer(y, h, s):
            return lmmse_equalizer(y, h, s + self.jammer_covariance, whiten_interference)

        super().__init__(equalizer=equalizer,
                         resource_grid=resource_grid,
                         stream_management=stream_management,
                         dtype=dtype, **kwargs)

    def set_jammer(self, j, rho):
        """
        Set the jammer covariance matrix from the frequency response of the jammer channel.

        Input
        -----
        j : [batch_size, num_rx, num_rx_ant, num_jammer, num_jammer_ant, num_ofdm_symbols, fft_size], tf.complex, frequency response of jammer channel
        rho : broadcastable to j.shape.
            Jammer power.
        """
        # scale j by rho
        j_dt = j * tf.sqrt(rho)
        # new shape: [batch_size, rx, num_sym, fft_size, rx_ant, tx*tx_ant]
        j_dt = tf.transpose(j_dt, perm=[0, 1, 5, 6, 2, 3, 4])
        j_dt = flatten_last_dims(j_dt, 2)
        j_dt = tf.cast(j_dt, dtype=self._dtype)
        
        # [batch_size, num_rx, num_ofdm_symbols, fft_size, num_rx_ant, num_rx_ant]
        self.jammer_covariance = tf.matmul(j_dt, j_dt, adjoint_b=True)

    def set_jammer_covariance(self, jammer_covariance):
        """
        Set the jammer covariance matrix directly.
        
        Input
        -----
        jammer_covariance: [batch_size, num_rx, num_ofdm_symbols, fft_size, num_rx_ant, num_rx_ant], tf.complex
            Covariance matrix of jammer signal.
        """
        self.jammer_covariance = jammer_covariance
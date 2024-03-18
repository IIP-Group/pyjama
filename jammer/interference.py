import sionna
import tensorflow as tf
from sionna.ofdm import LinearInterpolator

class CovarianceEstimator(tf.keras.Layer):
    r"""CovarianceEstimator(pilot_pattern)

    Parameters
    ----------
    pilot_pattern : PilotPattern
        An instance of :class:`~sionna.ofdm.PilotPattern`

    Input
    -----
    y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, num_effective_subcarriers], tf.complex

    Output
    ------
    R : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_rx_ant], tf.complex

    """
    def __init__(pilot_pattern):
        super(CovarianceEstimator, self).__init__()
        self._pilot_pattern = pilot_pattern
        # TODO don't do it with sionnas interpolators:
        # abuse of signature & inefficient  (over whole RG)
        # self._interpolator = LinearInterpolator()
        
        # find indices where all streams are masked
        self._estimation_indices = tf.where(tf.reduce_all(self._pilot_pattern.mask, axis=(0, 1)))
        
    def call(self, inputs):
        pass
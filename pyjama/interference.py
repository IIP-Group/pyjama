import sionna
import tensorflow as tf
import numpy as np

class CovarianceEstimator(tf.keras.layers.Layer):
    r"""CovarianceEstimator(pilot_pattern)

    Parameters
    ----------
    pilot_pattern : PilotPattern
        An instance of :class:`~sionna.ofdm.PilotPattern`.

    Input
    -----
    y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, num_effective_subcarriers], tf.complex

    Output
    ------
    R : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_rx_ant], tf.complex

    """
    def __init__(self, pilot_pattern, subcarrier_interpolation='nn'):
        super(CovarianceEstimator, self).__init__()
        self._pilot_pattern = pilot_pattern
        assert subcarrier_interpolation in ['nn']
        self._subcarrier_interpolation = subcarrier_interpolation

        # 1 where a pilot is zero, else 0
        zero_pilots = np.zeros_like(pilot_pattern.mask, dtype=np.complex64)
        zero_pilots[np.where(pilot_pattern.mask)] = np.reshape(np.abs(pilot_pattern.pilots)==0, -1)
        # num_estimation_elements, 2: symbol-index, subcarrier-index
        self._estimation_indices = tf.where(np.all(zero_pilots, axis=(0, 1)))
        if subcarrier_interpolation == 'nn':
            estimated_subcarriers = np.unique(self._estimation_indices[:, 1])
            # for each subcarrier, find the closest estimated subcarrier index
            closest_subcarrier_index = np.array([np.argmin(np.abs(estimated_subcarriers - subcarrier))
                                                for subcarrier in np.arange(pilot_pattern.num_effective_subcarriers)])
            self._closest_subcarrier = tf.gather(estimated_subcarriers, closest_subcarrier_index)
            

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        num_effective_subcarriers = input_shape[-1]
        # make RG dimensions first ones, to gather from them
        inputs = tf.transpose(inputs, tf.roll(tf.range(tf.rank(inputs)), shift=2, axis=0))
        # gather: [num_estimation_elements, batch_size, num_rx, num_rx_ant]
        estimation_signals = tf.gather_nd(inputs, self._estimation_indices)
        # compute element-wise covariances
        # [num_estimation_elements, batch_size, num_rx, num_rx_ant, num_rx_ant]
        cov_elementwise = tf.einsum('...i,...j->...ij', estimation_signals, tf.math.conj(estimation_signals))
        # mean over subcarrier using estimation_indices
        # [num_effective_subcarriers, batch_size, num_rx, num_rx_ant, num_rx_ant]
        # TODO change when upgrading TF, where tf.math.unsorted_mean support complex numbers
        mean_real = tf.math.unsorted_segment_mean(tf.math.real(cov_elementwise), self._estimation_indices[:, 1], num_effective_subcarriers)
        mean_imag = tf.math.unsorted_segment_mean(tf.math.imag(cov_elementwise), self._estimation_indices[:, 1], num_effective_subcarriers)
        cov_subcarrierwise = tf.complex(mean_real, mean_imag)
        # interpolate over subcarrier
        if self._subcarrier_interpolation == 'nn':
            cov_subcarrierwise = tf.gather(cov_subcarrierwise, self._closest_subcarrier, axis=0)
        # transpose to [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_rx_ant] and return
        cov = tf.transpose(cov_subcarrierwise, [1, 2, 0, 3, 4])
        cov = tf.expand_dims(cov, axis=2)
        output_shape = tf.concat([input_shape[:2], input_shape[3:], [input_shape[2]], [input_shape[2]]], axis=0)
        return tf.broadcast_to(cov, output_shape)
        
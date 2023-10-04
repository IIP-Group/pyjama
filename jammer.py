import numpy as np
import sionna
import tensorflow as tf


class OFDMJammer(tf.keras.layers.Layer):
    def __init__(self, channel, num_tx, num_tx_ant, max_amplitude, dtype=tf.complex64, **kwargs):
        super().__init__(trainable=False, dtype=dtype, **kwargs)
        self._channel = channel
        self._max_amplitude = max_amplitude
        self._num_tx = num_tx
        self._num_tx_ant = num_tx_ant
        
    def call(self, inputs):
        """First argument: unjammed signal. [batch_size, num_rx_ant, num_samples]]"""
        y_unjammed = inputs[0]
        input_shape = tf.shape(y_unjammed)

        jammer_input_shape = [input_shape[0], self._num_tx, self._num_tx_ant, input_shape[-2], input_shape[-1]]
        x_jammer = self.sample(jammer_input_shape)
        # noise is already added in the channel
        y_jammer = self._channel([x_jammer, 0.0])
        # in frequency domain we can just add the jammer signal
        y_combined = y_unjammed + y_jammer
        return y_combined
    
    def sample(self, shape):
        """Sample from complex plane with E[|x|^2] = 1]. In this case, we sample from uniform circle"""
        # TODO see above. Not done yet
        # sample from a uniform distribution with max amplitude
        dtype = tf.dtypes.as_dtype(self.dtype)
        if dtype.is_complex:
            return tf.complex(self._sample_real(shape, dtype.real_dtype), self._sample_real(shape, dtype.real_dtype))
        else:
            return self.sample_real(shape, self.dtype)
    
    def _sample_real(self, shape, dtype):
        return tf.random.uniform(shape, minval=-self._max_amplitude, maxval=self._max_amplitude, dtype=dtype)

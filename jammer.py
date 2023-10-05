import numpy as np
import sionna
from sionna.channel.ofdm_channel import OFDMChannel
import tensorflow as tf


class OFDMJammer(tf.keras.layers.Layer):
    def __init__(self, channel_model, rg, num_tx, num_tx_ant, jammer_power, normalize_channel=False, return_channel=False, dtype=tf.complex64, **kwargs):
        r"""jammer_power: NOT in dB, but "linear" power (i.e. 1.0 is 0 dB, 2.0 is 3 dB, etc.)"""
        super().__init__(trainable=False, dtype=dtype, **kwargs)
        self._channel_model = channel_model
        self._rg = rg
        self._num_tx = num_tx
        self._num_tx_ant = num_tx_ant
        self._jammer_power = jammer_power
        self._normalize_channel = normalize_channel
        self._return_channel = return_channel

    def build(self, input_shape):
        self._ofdm_channel = OFDMChannel(channel_model=self._channel_model,
                                         resource_grid=self._rg,
                                         add_awgn=False, # noise is already added in the ut-bs-channel
                                         normalize_channel=self._normalize_channel,
                                         return_channel=self._return_channel,
                                         dtype=tf.as_dtype(self.dtype))
        
    def call(self, inputs):
        """First argument: unjammed signal. [batch_size, num_rx_ant, num_samples]]"""
        y_unjammed = inputs[0]
        input_shape = tf.shape(y_unjammed)

        jammer_input_shape = [input_shape[0], self._num_tx, self._num_tx_ant, input_shape[-2], input_shape[-1]]
        x_jammer = self._jammer_power * self.sample(jammer_input_shape)
        if self._return_channel:
            y_jammer, h_freq_jammer = self._ofdm_channel(x_jammer)
        else:
            y_jammer = self._ofdm_channel(x_jammer)
        # in frequency domain we can just add the jammer signal
        y_combined = y_unjammed + y_jammer
        if self._return_channel:
            return y_combined, h_freq_jammer
        else:
            return y_combined
    
    def sample(self, shape):
        # sample from a unit disk in the complex plane with E[|x|^2] = 1
        dtype = tf.dtypes.as_dtype(self.dtype)
        if dtype.is_complex:
            # return tf.complex(self._sample_real(shape, dtype.real_dtype), self._sample_real(shape, dtype.real_dtype))
            return self._sample_complex(shape, dtype)
        else:
            raise TypeError("dtype must be complex")
    
    def _sample_complex(self, shape, dtype):
        """Sample from complex plane with E[|x|^2] = 1]. In this case, we sample from uniform circle.
        Sample theta and R uniform, r = sqrt(2R)"""
        r = tf.complex(tf.random.uniform(shape, minval=0, maxval=1, dtype=dtype.real_dtype), tf.cast(0.0, dtype.real_dtype))
        theta = tf.complex(tf.random.uniform(shape, minval=0, maxval=2*np.pi, dtype=dtype.real_dtype), tf.cast(0.0, dtype.real_dtype))
        return tf.sqrt(2*r)*tf.exp(1j*theta)

import numpy as np
import sionna
from sionna.channel.ofdm_channel import OFDMChannel
import tensorflow as tf
import copy
from .utils import _sample_complex_uniform_disk, _sample_complex_gaussian, _constellation_to_sampler

class OFDMJammer(tf.keras.layers.Layer):
    def __init__(self, channel_model, rg, num_tx, num_tx_ant, jammer_power, normalize_channel=False, return_channel=False, sampler="uniform", dtype=tf.complex64, **kwargs):
        r"""
        jammer_power: NOT in dB, but "linear" power (i.e. 1.0 is 0 dB, 2.0 is 3 dB, etc.)
        sampler: String in ["uniform", "gaussian"], a constellation, or function with signature (shape, dtype) -> tf.Tensor, where elementwise E[|x|^2] = 1
        """
        super().__init__(trainable=False, dtype=dtype, **kwargs)
        self._channel_model = channel_model
        self._rg = rg
        self._num_tx = num_tx
        self._num_tx_ant = num_tx_ant
        self._jammer_power = jammer_power
        self._normalize_channel = normalize_channel
        self._return_channel = return_channel
        self._dtype_as_dtype = tf.as_dtype(self.dtype)
        # if sampler is string, we use the corresponding function. Otherwise assign the function directly
        if isinstance(sampler, str):
            if sampler == "uniform":
                self._sample_function = _sample_complex_uniform_disk
            elif sampler == "gaussian":
                self._sample_function = _sample_complex_gaussian
            else:
                raise ValueError(f"Unknown sampler {sampler}")
        elif isinstance(sampler, sionna.mapping.Constellation):
            self._sample_function = _constellation_to_sampler(sampler, dtype=self._dtype_as_dtype)
        else:
            self._sample_function = sampler

    def build(self, input_shape):
        self._ofdm_channel = OFDMChannel(channel_model=self._channel_model,
                                         resource_grid=self._rg,
                                         add_awgn=False, # noise is already added in the ut-bs-channel
                                         normalize_channel=self._normalize_channel,
                                         return_channel=self._return_channel,
                                         dtype=self._dtype_as_dtype)
        
    def call(self, inputs):
        """First argument: unjammed signal. [batch_size, num_rx_ant, num_samples]]"""
        y_unjammed = inputs[0]
        # input_shape = y_unjammed.shape.as_list()
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
        
        if self._dtype_as_dtype.is_complex:
            return self._sample_function(shape, self._dtype_as_dtype)
        else:
            raise TypeError("dtype must be complex")
    
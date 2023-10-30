import numpy as np
import sionna
from sionna.channel.ofdm_channel import OFDMChannel
from sionna.ofdm import OFDMModulator, OFDMDemodulator
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel, time_lag_discrete_time_channel, time_to_ofdm_channel
from sionna.channel import ApplyTimeChannel, TimeChannel
import tensorflow as tf
import copy
from .utils import sample_function, ofdm_frequency_response_from_cir

class OFDMJammer(tf.keras.layers.Layer):
    def __init__(self, channel_model, rg, num_tx, num_tx_ant, normalize_channel=False, return_channel=False, sampler="uniform", dtype=tf.complex64, **kwargs):
        r"""
        sampler: String in ["uniform", "gaussian"], a constellation, or function with signature (shape, dtype) -> tf.Tensor, where elementwise E[|x|^2] = 1
        """
        super().__init__(trainable=False, dtype=dtype, **kwargs)
        self._channel_model = channel_model
        self._rg = rg
        self._num_tx = num_tx
        self._num_tx_ant = num_tx_ant
        self._normalize_channel = normalize_channel
        self._return_channel = return_channel
        self._dtype_as_dtype = tf.as_dtype(self.dtype)
        # if sampler is string, we use the corresponding function. Otherwise assign the function directly
        self._sample_function = sample_function(sampler, self._dtype_as_dtype)

    def build(self, input_shape):
        self._ofdm_channel = OFDMChannel(channel_model=self._channel_model,
                                         resource_grid=self._rg,
                                         add_awgn=False, # noise is already added in the ut-bs-channel
                                         normalize_channel=self._normalize_channel,
                                         return_channel=self._return_channel,
                                         dtype=self._dtype_as_dtype)
        
    def call(self, inputs):
        """First argument: unjammed signal. y: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
        Second argument: rho: broadcastable to [batch_size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]. Variances of jammer input signal (before channel)."""
        y_unjammed, rho = inputs
        # input_shape = y_unjammed.shape.as_list()
        input_shape = tf.shape(y_unjammed)

        jammer_input_shape = [input_shape[0], self._num_tx, self._num_tx_ant, input_shape[-2], input_shape[-1]]
        x_jammer = tf.sqrt(rho) * self.sample(jammer_input_shape)
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
    

class TimeDomainOFDMJammer(tf.keras.layers.Layer):
    def __init__(self, channel_model, rg, num_tx, num_tx_ant, send_cyclic_prefix=False, normalize_channel=False, return_channel=False, sampler="uniform", return_domain="freq", dtype=tf.complex64, **kwargs):
        """return_in_time_domain: One of ["freq", "time"]. Returns jammed signal in freqency or time domain. If return_channel is true, this might also be a pair of (signal, channel). Broadcast if not a pair in this case."""
        super().__init__(trainable=False, dtype=dtype, **kwargs)
        self._channel_model = channel_model
        self._rg = rg
        self._num_tx = num_tx
        self._num_tx_ant = num_tx_ant
        self._send_cyclic_prefix = send_cyclic_prefix
        self._normalize_channel = normalize_channel
        self._return_channel = return_channel
        if self._return_channel and len(return_domain) != 2:
            self._return_domain = (return_domain, return_domain)
        else:
            self._return_domain = return_domain
        self._dtype_as_dtype = tf.as_dtype(self.dtype)
        self._sampler = sample_function(sampler, self._dtype_as_dtype)

        self._l_min, self._l_max = time_lag_discrete_time_channel(self._rg.bandwidth)
        self._l_tot = self._l_max - self._l_min + 1
        self._channel_time = ApplyTimeChannel(self._rg.num_time_samples,
                                              l_tot=self._l_tot,
                                              add_awgn=False,
                                              dtype=self._dtype_as_dtype)
        self._modulator = OFDMModulator(rg.cyclic_prefix_length)
        self._demodulator = OFDMDemodulator(rg._fft_size, self._l_min, rg.cyclic_prefix_length)
        
    def __call__(self, inputs):
        """
        Input: Signal in time domain.
        Output: Jammed signal in time domain or frequency domain according to return_in_time_domain.
        First argument: unjammed signal in time domain. y: [batch_size, num_rx, num_rx_ant, num_time_samples + l_max - l_min]
        Second argument: rho: broadcastable to [batch_size, num_tx, num_tx_ant, num_time_samples]. Variances of jammer input signal (before channel)."""
        y_time, rho = inputs
        batch_size = tf.shape(y_time)[0]
        a, tau = self._channel_model(batch_size, self._rg.num_time_samples + self._l_tot - 1, self._rg.bandwidth)
        h_time = cir_to_time_channel(self._rg.bandwidth, a, tau, self._l_min, self._l_max, self._normalize_channel)

        if self._send_cyclic_prefix:
            x_jammer_freq = self._sampler([batch_size, self._num_tx, self._num_tx_ant, self._rg.num_ofdm_symbols, self._rg.fft_size], self._dtype_as_dtype)
            x_jammer_time = self._modulator(x_jammer_freq)
        else:
            x_jammer_time = self._sampler([batch_size, self._num_tx, self._num_tx_ant, self._rg.num_ofdm_symbols * (self._rg.fft_size + self._rg.cyclic_prefix_length)], self._dtype_as_dtype)
        
        x_jammer_time = tf.sqrt(rho) * x_jammer_time
        y_time = y_time + self._channel_time([x_jammer_time, h_time])

        if self._return_channel:
            y_ret = y_time if self._return_domain[0] == "time" else self._demodulator(y_time)
            # h_ret = h_time if self._return_in_time_domain[1] else ofdm_frequency_response_from_cir(a, tau, self._rg, normalize=self._normalize_channel)
            h_ret = h_time if self._return_domain[1] == "time" else time_to_ofdm_channel(h_time, self._rg, self._l_min)
            return y_ret, h_ret
        else:
            return y_time if self._return_domain == "time" else self._demodulator(y_time)
        # if self._return_in_time_domain:
        #     if self._return_channel:
        #         return y_time, h_time
        #     else:
        #         return y_time
        # else:
        #     y_freq = self._demodulator(y_time)
        #     if self._return_channel:
        #         # We need to downsample the path gains `a` to the OFDM symbol rate prior to converting the CIR to the channel frequency response.
        #         h_freq = ofdm_frequency_response_from_cir(a, tau, self._rg, normalize=self._normalize_channel)
        #         return y_freq, h_freq
        #     else:
        #         return y_freq
        
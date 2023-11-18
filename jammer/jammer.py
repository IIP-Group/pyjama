import numpy as np
import sionna
from sionna.channel.ofdm_channel import OFDMChannel
from sionna.ofdm import OFDMModulator, OFDMDemodulator
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel, time_lag_discrete_time_channel, time_to_ofdm_channel
from sionna.channel import ApplyTimeChannel, TimeChannel
import tensorflow as tf
import copy
from .utils import sample_function, NonNegMaxSquareNorm

class OFDMJammer(tf.keras.layers.Layer):
    def __init__(self, channel_model, rg, num_tx, num_tx_ant, jamming_type="barrage", density_symbols=1.0, density_subcarriers=1.0, normalize_channel=False, return_channel=False, sampler="uniform", trainable=False, trainable_mask=None, dtype=tf.complex64, **kwargs):
        r"""
        sampler: String in ["uniform", "gaussian"], a constellation, or function with signature (shape, dtype) -> tf.Tensor, where elementwise E[|x|^2] = 1
        trainable_mask: boolean, shape broadcastable to jammer_input_shape. If True, the corresponding element is trainable. If False, the corresponding element is held constant (jammer_power power). If None, all elements are trainable.
        """
        super().__init__(trainable=trainable, dtype=dtype, **kwargs)
        self._channel_model = channel_model
        self._rg = rg
        self._num_tx = num_tx
        self._num_tx_ant = num_tx_ant
        self._jamming_type = jamming_type
        self._density_symbols = density_symbols
        self._density_subcarriers = density_subcarriers
        self._normalize_channel = normalize_channel
        self._return_channel = return_channel
        self._dtype_as_dtype = tf.as_dtype(self.dtype)
        # if sampler is string, we use the corresponding function. Otherwise assign the function directly
        self._sample_function = sample_function(sampler, self._dtype_as_dtype)
        self._ofdm_channel = OFDMChannel(channel_model=self._channel_model,
                                         resource_grid=self._rg,
                                         add_awgn=False, # noise is already added in the ut-bs-channel
                                         normalize_channel=self._normalize_channel,
                                         return_channel=self._return_channel,
                                         dtype=self._dtype_as_dtype)
        self._check_settings()

    def _check_settings(self):
        assert self._jamming_type in ["barrage", "pilot", "data"], "jamming_type must be one of ['barrage', 'pilot', 'data']"
        assert self._density_symbols >= 0.0 and self._density_symbols <= 1.0, "density_symbols must be in [0, 1]"
        assert self._density_subcarriers >= 0.0 and self._density_subcarriers <= 1.0, "density_subcarriers must be in [0, 1]"
        if self._jamming_type in ["pilot", "data"]:
            # TODO: pilot and data jamming not very well defined for sparse jamming. Discuss.
            assert self._density_symbols == 1.0, "density_symbols must be 1.0 for jamming_type 'pilot' or 'data'"
            assert self._density_subcarriers == 1.0, "density_subcarriers must be 1.0 for jamming_type 'pilot' or 'data'"
            
    def build(self, input_shape):
        # TODO: implement trainable_mask as described in docstring
        # below: weights only over ofdm_symbols
        num_ofdm_symbols = input_shape[0][-2]
        constraint = NonNegMaxSquareNorm(num_ofdm_symbols)
        self._weights = tf.Variable(tf.ones([num_ofdm_symbols, 1]), dtype=self._dtype_as_dtype.real_dtype, trainable=self.trainable, constraint=constraint)
        
    def call(self, inputs):
        """First argument: unjammed signal. y: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
        Second argument: rho: broadcastable to [batch_size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]. Variances of jammer input signal (before channel)."""
        y_unjammed, rho = inputs
        input_shape = tf.shape(y_unjammed)

        # jammer_input_shape = [input_shape[0], self._num_tx, self._num_tx_ant, input_shape[-2], input_shape[-1]]
        jammer_input_shape = tf.concat([[input_shape[0]], [self._num_tx, self._num_tx_ant], input_shape[-2:]], axis=0)
        x_jammer = self.sample(jammer_input_shape)
        rho = self.make_sparse(rho, tf.shape(x_jammer))
        x_jammer = tf.sqrt(rho) * x_jammer
        x_jammer = tf.cast(self._weights, x_jammer.dtype) * x_jammer
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

    def make_sparse(self, data, shape):
        """Returns data broadcasted to shape, where s_symbol*num_symbols symbols and s_subcarrier*num_subcarriers subcarriers are non-zero"""
        data = tf.broadcast_to(data, shape)

        if self._jamming_type == "barrage":
            num_symbols, num_subcarriers = shape[-2], shape[-1]
            num_nonzero_symbols = tf.cast(tf.round(self._density_symbols * tf.cast(num_symbols, tf.float32)), tf.int32)
            num_nonzero_subcarriers = tf.cast(tf.round(self._density_subcarriers * tf.cast(num_subcarriers, tf.float32)), tf.int32)

            # create sparse masks
            symbol_mask = tf.concat([tf.ones([num_nonzero_symbols]), tf.zeros([num_symbols - num_nonzero_symbols])], axis=0)
            symbol_mask = tf.random.shuffle(symbol_mask)
            subcarrier_mask = tf.concat([tf.ones([num_nonzero_subcarriers]), tf.zeros([num_subcarriers - num_nonzero_subcarriers])], axis=0)
            subcarrier_mask = tf.random.shuffle(subcarrier_mask)

            return data * tf.cast(tf.matmul(symbol_mask[...,tf.newaxis], subcarrier_mask[...,tf.newaxis], transpose_b=True), data.dtype)
        
        # only pilot or data
        pilot_mask = tf.cast(self._rg.pilot_pattern.mask, tf.bool)
        # take mask where any UT is transmitting, i.e. sum over all (tx, tx_ant) dimensions
        pilot_mask = tf.reduce_any(pilot_mask, axis=[0, 1])
        
        if self._jamming_type == "pilot":
            return data * tf.cast(pilot_mask, data.dtype)
        else:
            return data * tf.cast(tf.logical_not(pilot_mask), data.dtype)



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

        # TODO only for 802.11n
        self._l_min, self._l_max = time_lag_discrete_time_channel(self._rg.bandwidth)
        # self._l_min, self._l_max = time_lag_discrete_time_channel(self._rg.bandwidth, maximum_delay_spread=2.0e-6)
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
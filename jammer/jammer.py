"""
This module provides classes for simulating one or multiple jammers.
They are meant to be used to simulate jammers in the context of OFDM-based communication.

The main way of functioning is the following:

|  1. Build a communication pipeline between "regular" UEs and BSs.
|  2. Instantiate an additional channel model between the jammers and the BSs.
|  3. Instantiate a jammer, using the channel created in step 2.
|  4. Add the jammer to the communication pipeline, right after the channel between the UEs and the BSs like so:
>>> y = channel([x, no])
>>> y_jammed = jammer([y, rho])


The module consists of two main classes: :class:`OFDMJammer` and :class:`TimeDomainOFDMJammer`.
The former is to be used for simulations solely in the frequency domain, while the latter is to be used for simulations in the time domain.

`TimeDomainOFDMJammer` can hence be used to jammers which violate the OFDM assumptions, i.e. not sending a cyclic prefix. The simulation is much slower, however.

`OFDMJammer` currently has much more functionality implemented, and can i.e. be used to simulate learning/smart jammers.

"""

import numpy as np
import sionna
from sionna.channel.ofdm_channel import OFDMChannel
from sionna.ofdm import OFDMModulator, OFDMDemodulator
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel, time_lag_discrete_time_channel, time_to_ofdm_channel
from sionna.channel import ApplyTimeChannel, TimeChannel
import tensorflow as tf
import copy
from .utils import sample_function, NonNegMaxMeanSquareNorm, MaxMeanSquareNorm

# TODO add the complete formula into docstring (with rho, jammer_weights, jammer symbols, etc.). See sionna.channel.ofdm_channel.OFDMChannel for reference.
class OFDMJammer(tf.keras.layers.Layer):
    r"""
    Used to simulate a jammer in the frequency domain only (i.e. sending a cyclic prefix). The jammer signal is generated through the given (Jammer-BS) channel model.

    It can be trained, and support several other functionalities, which are described in the following.
    
    Jamming Types
        Jamming types refer to which resource elements are jammed. This can additionally be modified by the shape of :math:`\rho` (see below).
        The following jamming types are supported:

        - Barrage Jamming: All resource elements are jammed.
        - Pilot Jamming: Only pilot resource elements are jammed.
        - Data Jamming: Only data resource elements are jammed.
        - Non-Silent Jamming: Only non-silent resource elements are jammed. This is similar to pilot jamming, but does not jam the "silent" pilot symbols where jammer usually is estimated.

        The power of the symbols sent by the jammer will be scaled so that the mean power of each resource element over the whole resource grid is equal to :math:`\rho`.

    Sparse Jamming
        Sparse jamming refers to the fact that only a fraction of the resource elements are jammed. This is controlled by the parameters `density_symbols` and `density_subcarriers`.
        A value of 1.0 means that all resource elements are jammed, while a value of 0.0 means that no resource elements are jammed.

        This is done by nulling said percentage of rows or columns of the resource grid. Hence, it is meant to be understood not as an exact number of resource elements, but rounded to the next integer.
        
        Sparsity is ill-defined for non-barrage jammers. For example, pilots might not be evenly distributed. It would then not be clear what sparsity means. Hence, sparsity is only supported for barrage jammers.

    Jamming Power & Jamming Pattern
        The jamming power is controlled by the parameter :math:`\rho`. It is given for every call of the jammer, and can be thought of as the square of coefficients which the jammer symbols are multiplied by.
        It might be a single value, or a tensor of shape broadcastable to `[batch_size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]`, where `num_tx` and `num_tx_ant` are the number of jammers and their antennas, respectively.
        
        One can hence e.g. jam only a part of the resource grid by using passing in a tensor of shape `[num_ofdm_symbols, fft_size]`, where several resource elements are zero,
        or regulate the power of different jammers by passing in a tensor of shape `[num_tx, 1, 1, 1]`, with different values for each jammer.

    Different Sampling Methods
        The jammers can be thought of as UEs, which sample their symbols :math:`x` somehow with power :math:`E[|x|^2] = 1`. These symbols are then scaled by :math:`\sqrt{\rho}`, possibly nulled (see above) and sent through the jammer channel.
        The result is then added to the received signal.
        
        The interface of this class accepts a number of strings for predefined sampling methods. Alternatively, a constellation which is randomly sampled
        or a callable which acts as sampling function can be passed in. See documentation of the input parameter `sampler` for the specific types.

    Training
        In general, the jammer can be trained by setting `trainable=True`. Weights (i.e. coefficients of the sampled jammer symbols) can then be learned.
        
        As parameters to an instance of this class, one can pass a mask, a constraint and a boolean `constraint_integrated`.
        The mask is a boolean tensor of shape broadcastable to `[num_tx, num_tx_ant, num_ofdm_symbols, fft_size]`.
        If `True`, the corresponding weight is trainable. If `False`, the weight is held constant at 1.0.

        To all trainable weights, a constraint is applied. By default, this constrains the mean power of those weights to be :math:`\leq 1.0`.

        This constraint can either be applied after each step of the optimizer, or part of the calculation itself. This is controlled by the boolean `constraint_integrated`.
        Empirically speaking, the former might help breaking out of local minima (as the optimizer just takes a step in direction of the unconstrained optimum, the constraint is just applied afterwards)
        while the latter helps with convergence (as the constraint is reflected in the gradient).

    Parameters
    ----------
    channel_model: ChannelModel
        Instance of :class:`sionna.channel.ChannelModel`.Channel between jammer(s) and BS(s).
    rg: ResourceGrid
        Instance of :class:`sionna.ofdm.ResourceGrid`. Resource grid of the OFDM system.
    num_tx: int
        Number of jammers.
    num_tx_ant: int
        Number of antennas of each jammer.
    jamming_type: str
        One of ["barrage", "pilot", "data", "non_silent"].
    density_symbols: float
        Fraction of symbol times which are jammed. Must be in between 0 and 1.
    density_subcarriers: float
        Fraction of subcarriers which are jammed. Must be in between 0 and 1.
    normalize_channel: bool
        Whether to normalize the channel. If True, the channel is normalized so that for each link the mean energy of each channel coefficient is 1.0.
        (this is just for the channel coefficients).
    return_channel: bool
        Whether to return the jammer channel frequency response should be returned in addition to the jammed signal.
    sampler: str, instance of :class:`~sionna.mapping.Constellation`, or callable
        If str, one of ["uniform", "gaussian"].
        If instance of :class:`~sionna.mapping.Constellation`, the constellation is sampled randomly.
        If callable, the callable is used as sampling function. It should have signature ``(shape, dtype) -> tf.Tensor``.
        In the first two cases, the mean energy of the sampled symbols is 1.0. In the last case, the caller is responsible for the energy of the sampled symbols.
    trainable: bool
        Whether the jammer weights are trainable.
    trainable_mask: None, or tensor broadcastable to `[num_tx, num_tx_ant, num_ofdm_symbols, fft_size]`
        If None, all weights are trainable. If tensor, only weights where the mask is True are trainable.
    training_constraint: None, callable or instance of :class:`tf.keras.constraints.Constraint`
        Constraint to be applied to the trainable weights.
        If None, no constraint is applied.
        The callable should take a tensor as argument and return a tensor of the same shape.
    constraint_integrated: bool
        If True, the constraint is integrated into the network. If False, the constraint is applied after each optimization step.
    dtype: tf.complex
        Data type of the jammer symbols. Defaults to tf.complex64.

    Input
    -----
    y: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex
        Unjammed signal. Shape 
    rho: broadcastable to [batch_size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.float
        Power of jammer signal. See section "Jamming Power & Jamming Pattern" above for details.

    Output
    ------
    (y_jammed, h_freq_jammer) or y_jammed : Tuple or Tensor
    y_jammed: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex
        Input signal with jammer interference added.
    h_freq_jammer: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.complex
        Frequency response of jammer channel. Only returned if ``return_channel`` is True.
    
    """
    def __init__(self,
                 channel_model,
                 rg,
                 num_tx,
                 num_tx_ant,
                 jamming_type="barrage",
                 density_symbols=1.0,
                 density_subcarriers=1.0,
                 normalize_channel=True,
                 return_channel=False,
                 sampler="uniform",
                 trainable=False,
                 trainable_mask=None,
                 training_constraint=MaxMeanSquareNorm(1.0),
                # TODO is this a good name for this parameter?
                 constraint_integrated=True,
                 dtype=tf.complex64,
                 **kwargs):
        r"""
        sampler: String in ["uniform", "gaussian"], a constellation, or function with signature (shape, dtype) -> tf.Tensor, where elementwise E[|x|^2] = 1
        trainable_mask: boolean, shape broadcastable to jammer_input_shape. If True, the corresponding element is trainable. If False, the corresponding element is held constant (jammer_power power). If None, all elements are trainable.
        constraint_integrated: boolean. If True, the constraint is integrated into the network. If False, the constraint is applied after each optimization step.
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
        self._trainable_mask = trainable_mask
        self._training_constraint = training_constraint
        self._constraint_integrated = constraint_integrated
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
        assert self._jamming_type in ["barrage", "pilot", "data", "non_silent"], "jamming_type must be one of ['barrage', 'pilot', 'data']"
        # TODO if non_silent, check that outer pilot pattern is PilotPatternWithSilence
        assert self._density_symbols >= 0.0 and self._density_symbols <= 1.0, "density_symbols must be in [0, 1]"
        assert self._density_subcarriers >= 0.0 and self._density_subcarriers <= 1.0, "density_subcarriers must be in [0, 1]"
        if self._jamming_type in ["pilot", "data"]:
            # TODO: pilot and data jamming not very well defined for sparse jamming. Discuss.
            assert self._density_symbols == 1.0, "density_symbols must be 1.0 for jamming_type 'pilot' or 'data'"
            assert self._density_subcarriers == 1.0, "density_subcarriers must be 1.0 for jamming_type 'pilot' or 'data'"
            
    def build(self, input_shape):
        if self._trainable_mask is None:
            # all weights are trainable
            jammer_input_shape = tf.concat([[self._num_tx, self._num_tx_ant], input_shape[0][-2:]], axis=0)
            self._trainable_mask = tf.ones(jammer_input_shape, dtype=tf.bool)

        self._training_indices = tf.where(self._trainable_mask)
        count_trainable = tf.shape(self._training_indices)[0]
        if self.trainable:
            constraint = None if self._constraint_integrated else self._training_constraint
            self._training_weights = tf.Variable(tf.random.uniform([count_trainable], minval=0.8, maxval=1.0),
                                                 dtype=self._dtype_as_dtype.real_dtype, trainable=True, constraint=constraint)
        else:
            self._training_weights = tf.Variable(tf.ones([count_trainable]), dtype=self._dtype_as_dtype.real_dtype, trainable=False)
            
        self._weights = tf.Variable(tf.ones(self._trainable_mask.shape), trainable=False)
        
    def call(self, inputs):
        """First argument: unjammed signal. y: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
        Second argument: rho: broadcastable to [batch_size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]. Variances of jammer input signal (before channel)."""
        y_unjammed, rho = inputs
        rho = tf.cast(rho, self._dtype)
        input_shape = tf.shape(y_unjammed)

        # jammer_input_shape = [input_shape[0], self._num_tx, self._num_tx_ant, input_shape[-2], input_shape[-1]]
        jammer_input_shape = tf.concat([[input_shape[0]], [self._num_tx, self._num_tx_ant], input_shape[-2:]], axis=0)
        x_jammer = self._sample(jammer_input_shape)

        # TODO check interaction with rho
        # weights have mean(|w|^2) <= 1
        if self._constraint_integrated:
            constrained_training_weights = self._training_constraint(self._training_weights)
        else:
            constrained_training_weights = self._training_weights
        weights = tf.tensor_scatter_nd_update(tf.ones(self._trainable_mask.shape), self._training_indices, constrained_training_weights)
        self._weights.assign(weights)

        x_jammer = tf.cast(weights, x_jammer.dtype) * x_jammer
        rho = self._make_sparse(rho, tf.shape(x_jammer))

        x_jammer = tf.sqrt(rho) * x_jammer
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
    
    def _sample(self, shape):
        if self._dtype_as_dtype.is_complex:
            return self._sample_function(shape, self._dtype_as_dtype)
        else:
            raise TypeError("dtype must be complex")

    def _make_sparse(self, data, shape):
        """Returns data broadcasted to shape, where s_symbol*num_symbols symbols and s_subcarrier*num_subcarriers subcarriers are non-zero.
        Data is scaled so that the mean power of the output equals the mean power of the input.
        Data is assumed to be power (not amplitude)."""
        # the meaning of the sparsity parameters with non-barrage jammers is not clear. For now, the sparsity parameters are only permitted for barrage jammers.
        if self._jamming_type != "barrage":
            assert self._density_symbols == 1.0, "density_symbols must be 1.0 for any jamming_type other than 'barrage'"
            assert self._density_subcarriers == 1.0, "density_subcarriers must be 1.0 for any jamming_type other than 'barrage'"

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

            sparsity_mask = tf.cast(tf.matmul(symbol_mask[...,tf.newaxis], subcarrier_mask[...,tf.newaxis], transpose_b=True), data.dtype)
        else:
            # only pilot or data
            pilot_mask = tf.cast(self._rg.pilot_pattern.mask, tf.bool)
            # take mask where any UT is transmitting, i.e. sum over all (tx, tx_ant) dimensions
            pilot_mask = tf.reduce_any(pilot_mask, axis=[0, 1])
        
            if self._jamming_type == "pilot":
                sparsity_mask = tf.cast(pilot_mask, data.dtype)
            elif self._jamming_type == "data":
                sparsity_mask = tf.cast(tf.logical_not(pilot_mask), data.dtype)
            elif self._jamming_type == "non_silent":
                # TODO make internal_pilot_mask a property of the pilot pattern
                internal_pilot_mask = tf.cast(self._rg.pilot_pattern._internal_pilot_pattern.mask, tf.bool)
                internal_pilot_mask = tf.reduce_any(internal_pilot_mask, axis=[0, 1])
                silent_mask = pilot_mask & tf.logical_not(internal_pilot_mask)
                sparsity_mask = tf.cast(tf.logical_not(silent_mask), data.dtype)
            else:
                raise ValueError("jamming_type must be one of ['barrage', 'pilot', 'data', 'non_silent']")
        # scale rho to account for sparsity introduced by make_sparse. Sparsity of data should not contribute to this!
        output = data * sparsity_mask
        sparsity = tf.nn.zero_fraction(sparsity_mask)
        if sparsity < 1.0:
            scale = 1.0 / (1.0 - tf.cast(sparsity, output.dtype))
            output = output * scale
        # else, all elements are zero, so we don't need to scale
        return output



class TimeDomainOFDMJammer(tf.keras.layers.Layer):
    """
    This class is meant ot simulate jammers in the time domain channel. It is much slower than :class:`OFDMJammer`, but can be used to simulate jammers which violate the OFDM assumptions, i.e. not sending a cyclic prefix.
    It thus functions similarily to :class:`OFDMJammer`, with the following differences:
    
    - Some functionality is not supported, such as learning jammer weights. This might be implemented in the future.
    - The input signal is assumed to be in the time domain. The output can be chosen to be in the time domain or frequency domain using the parameter `return_domain`.
    - Cyclic prefix will or will not be sent depending on the parameter `send_cyclic_prefix`.

    Parameters
    ----------
    channel_model: ChannelModel
        Instance of :class:`sionna.channel.ChannelModel`.Channel between jammer(s) and BS(s).
    rg: ResourceGrid
        Instance of :class:`sionna.ofdm.ResourceGrid`. Resource grid of the OFDM system.
    num_tx: int
        Number of jammers.
    num_tx_ant: int
        Number of antennas of each jammer.
    send_cyclic_prefix: bool
        If true, the jammer adheres to the OFDM assumptions and sends a cyclic prefix. If false, the jammer sends randomly sampled symbols instead.
    normalize_channel: bool
        Whether to normalize the channel. If True, the channel is normalized so that for each link the mean energy of each channel coefficient is 1.0.
    return_channel: bool
        If true, 
    
    """
    def __init__(self,
                 channel_model,
                 rg,
                 num_tx,
                 num_tx_ant,
                 send_cyclic_prefix=False,
                 maximum_delay_spread=3e-6,
                 normalize_channel=False,
                 return_channel=False,
                 sampler="uniform",
                 return_domain="freq",
                 dtype=tf.complex64,
                 **kwargs):
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
        self._l_min, self._l_max = time_lag_discrete_time_channel(self._rg.bandwidth, maximum_delay_spread)
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

        # TODO is there an error here? Est. CSI Jammer Mitigation works better than perfect CSI Jammer Mitigation
        if self._return_channel:
            y_ret = y_time if self._return_domain[0] == "time" else self._demodulator(y_time)
            # h_ret = h_time if self._return_in_time_domain[1] else ofdm_frequency_response_from_cir(a, tau, self._rg, normalize=self._normalize_channel)
            h_ret = h_time if self._return_domain[1] == "time" else time_to_ofdm_channel(h_time, self._rg, self._l_min)
            return y_ret, h_ret
        else:
            return y_time if self._return_domain == "time" else self._demodulator(y_time)
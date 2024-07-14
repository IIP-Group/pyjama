#%%
import os
import sionna
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import math
import time
import datetime
import pickle

from sionna.mimo import StreamManagement

from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
from sionna.ofdm import OFDMModulator, OFDMDemodulator, ZFPrecoder, RemoveNulledSubcarriers

from sionna.channel.tr38901 import Antenna, AntennaArray, CDL, UMi, UMa, RMa
from sionna.channel import gen_single_sector_topology
from sionna.channel import cir_to_ofdm_channel, cir_to_time_channel, time_lag_discrete_time_channel
from sionna.channel import ApplyOFDMChannel, ApplyTimeChannel, OFDMChannel, RayleighBlockFading

from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder

from sionna.mapping import Mapper, Demapper

from sionna.utils import BinarySource, ebnodb2no, sim_ber, plot_ber, QAMSource, PlotBER
from sionna.utils.metrics import compute_ber

from .jammer import OFDMJammer, TimeDomainOFDMJammer
from .mitigation import POS, IAN
from .pilots import OneHotWithSilencePilotPattern, OneHotPilotPattern, PilotPatternWithSilence
from .channel_models import MultiTapRayleighBlockFading
from .utils import covariance_estimation_from_signals, linear_to_db, db_to_linear, plot_to_image, plot_matrix, matrix_to_image, reduce_mean_power, normalize_power#, expected_bitflips

from tensorflow.python.keras.losses import BinaryCrossentropy, MeanAbsoluteError, MeanSquaredError

# TODO find a better way than global variables
BATCH_SIZE = 2
MAX_MC_ITER = 30
EBN0_DB_MIN = -5.0
EBN0_DB_MAX = 15.0
NUM_SNR_POINTS = 16
ebno_dbs = np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, NUM_SNR_POINTS)
# TODO let the plot ylim always be 1.0
ber_plots = PlotBER()



# sionna.config.xla_compat=True
class Model(tf.keras.Model):
    """
    Model to simulate OFDM MIMO transmissions, mainly over a 3GPP 38.901 model.
    It supports different simulations in frequency and time domain, as well as simulation of communication with jammer(s) present.
    Several jammer mitigation algorithms are implemented as well.
    
    In all these simulations, trainable elements like jammers, receivers etc. are supported (only jammers are implemented so far).

    Parameters
    ----------
    scenario : str, optional
        The 3GPP 38.901 scenario to simulate. The default is "umi". Must be one of ["umi", "uma", "rma", "rayleigh", "multitap_rayleigh"].
    carrier_frequency : float, optional
        The carrier frequency in Hz. The default is 3.5e9.
    fft_size : int, optional
        The number of OFDM subcarriers. The default is 128.
    subcarrier_spacing : float, optional
        The subcarrier spacing in Hz. The default is 30e3.
    num_ofdm_symbols : int, optional
        The number of OFDM symbols per resource grid. The default is 14.
    cyclic_prefix_length : int, optional
        The length of the cyclic prefix in samples. The default is 20.
    maximum_delay_spread : float, optional
        The maximum delay spread in seconds. This parameter is only relevant for time domain simulations.  The default is 3e-6.
    num_bs_ant : int, optional
        The number antennas on the basestation. The default is 16.
    num_ut : int, optional
        The number of User Terminals (UTs) simulated. The default is 4.
    num_ut_ant : int, optional
        The number of antennas on each UT. The default is 1.
    num_bits_per_symbol : int, optional
        The number of bits per constellation symbol. The default is 2 (QPSK).
    coderate : float, optional
        The coderate of the LDPC code. The default is None, which means no coding.
    decoder_parameters : dict, optional
        Additional parameters for the LDPC decoder. By default, the encoder will be set, and hard_out will be set to False (i.e. soft output).
    domain : str, optional
        The domain in which the simulation takes place. The default is "freq". Must be one of ["freq", "time"].
    los : bool or None, optional
        If not None, all UTs are forced to be in line-of-sight (LOS) with the basestation or not.
    indoor_probability : float, optional
        Probability of a UT being indoors. The default is 0.8.
    min_ut_velocity : None or tf.float
        Minimum UT velocity [m/s]
    max_ut_velocity : None or tf.float
        Maximim UT velocity [m/s]
    resample_topology : bool, optional
        If True, the topology is resampled for each batch. The default is True.
    perfect_csi : bool, optional
        If True, the channel is assumed to be perfectly known at the receiver. The default is False.
    perfect_jammer_csi : bool, optional
        If True, the jammer-BS channel is assumed to be perfectly known at the receiver (for mitigation). The default is False.
    num_silent_pilot_symbols : int, optional
        The number of pilot symbols where no UT tranmits (can be used for jammer channel estimation). The default is 0.
    channel_parameters : dict, optional
        Additional parameters for the channel model. The default is {}.
    jammer_present : bool, optional
        If True, a jammer/jammers are present in the simulation. The default is False.
    jammer_power : :class:`tf.Tensor` broadcastable to [batch_size, num_jammer, num_jammer_ant, num_ofdm_symbols, fft_size], float; or float, optional
        Power of the jamming signal (of each jammer antenna). See :class:`~jammer.jammer.OFDMJammer` for details. The default is 1.0.
    jammer_parameters : dict, optional
        Additional parameters for the jammer, which is an instance of :class:`~jammer.jammer.OFDMJammer` or :class:`~jammer.jammer.TimeDomainOFDMJammer`.
        The default is {"num_tx": 1, "num_tx_ant": 1, "normalize_channel": True}.
    min_jammer_velocity : None or tf.float
        Minimum jammer velocity [m/s]
    max_jammer_velocity : None or tf.float
        Maximim jammer velocity [m/s]
    jammer_mitigation : str or None, optional
        The jammer mitigation algorithm to use. The default is None, i.e. the jammer will not be mitigated.
        If it is a string, it must be one of ["pos", "ian"]. See :doc:`mitigation` for details.
    jammer_mitigation_dimensionality : int or None, optional
        The dimensionality of the jammer mitigation. For each dimension, the largest remaining eigenvalue of the jammer covariance matrix is removed,
        at the cost of ~1 degree of freedom (depending on the mitigation strategy).
        If None, the dimensionality will be set to the rank of the jammer covariance matrix.
        I.e. the total number of jammer antennas transmitting (when perfect_jammer_csi is True) or
        the maximum of the number of jammer antennas transmitting in the silent pilot symbols and num_silent_pilot_symbols (when perfect_jammer_csi is False).
        The default is None.
    return_jammer_signals : bool, optional
        If True, the received jammer signals during the silent pilot symbols are return additionally. The default is False.
    return_symbols : bool, optional
        If True, the estimated symbols are returned instead of the estimated bit-LLRs. The default is False.
    return_decoder_iterations : bool, optional
        If True, instead of the final LLRs, the LLRs after each iteration of the LDPC decoder are returned. The default is False.
    
    Input
    -----
    batch_size : int
        The number of resource grids to simulate.
    ebno_db : float
        The SNR (Eb/N0) of the signal sent by the UTs in dB.
    training : bool, optional
        If True, the model is in training mode. If False, the model is in inference mode. The default is None, which means inference mode.
        Currently this only matters for the LDPC decoder, which returns intermediate results during training and the only the final result during inference.
    
    Output
    ------
    (b, llr) or (x, x_hat), or (b, llr, jammer_signals) or (x, x_hat, jammer_signals) :
        Either the transmitted bits and the estimated bit-LLRs, or the transmitted symbols and the estimated symbols (depending on ``return_symbols``).
        If ``return_jammer_signals`` is True, the jammer signals during the silent pilot symbols are returned as well.
    b : [batch_size, num_tx * num_streams_per_tx * k], tf.float32
        The transmitted bits.
    llr : [batch_size, num_tx * num_streams_per_tx * k] or [batch_size, num_tx * num_streams_per_tx * k * num_decoder_iterations], tf.float32
        The estimated bit-LLRs. If ``return_decoder_iterations`` is True, the LLRs after each iteration of the LDPC decoder are returned.
    x : [batch_size, num_tx * num_streams_per_tx * num_data_symbols], tf.complex64
        The transmitted symbols.
    x_hat : [batch_size, num_tx * num_streams_per_tx * num_data_symbols], tf.complex64
        The estimated symbols.
    jammer_signals : [batch_size, num_rx, num_rx_ant, num_silent_pilot_symbols, fft_size], tf.complex64
        The jammer signals during the silent pilot symbols.
    
    Note
    ----
    This simulation model uses the assumption that the jammer is not included in the noise estimation (only :math:N_0).
    This is a simplification, as the noise is normally estimated separately from the signal (and hence would include noise caused by the jammer).
    In consequence, the LLR estimation is over-confident, as the noise is underestimated.
    This is unproblematic for uncoded training (as only the sign of the LLR matters), but degrades the performance of the LDPC decoder.
    The stronger the jammer, the more the performance is degraded.
    
    """
    # TODO the dict-arguments could also be replaced by **kwargs. Deliberate.
    # see https://rhettinger.wordpress.com/2011/05/26/super-considered-super/ for reference.
    # TODO jammer should have a separate los and indoor_probability parameter
    # TODO make return_symbols property?
    def __init__(self,
                 scenario="umi",
                 carrier_frequency=3.5e9,
                 fft_size=128,
                 subcarrier_spacing=30e3,
                 num_ofdm_symbols=14,
                 cyclic_prefix_length=20,
                 maximum_delay_spread=3e-6,
                 num_bs_ant=16,
                 num_ut=4,
                 num_ut_ant=1,
                 num_bits_per_symbol=2,
                 coderate=None,
                 decoder_parameters={},
                 domain="freq",
                 los=None,
                 indoor_probability=0.8,
                 min_ut_velocity=0.0,
                 max_ut_velocity=0.0,
                 resample_topology=True,
                 perfect_csi=False,
                 perfect_jammer_csi=False,
                 num_silent_pilot_symbols=0,
                 channel_parameters={},
                 jammer_present=False,
                 jammer_power=1.0,
                 jammer_parameters={
                     "num_tx": 1,
                     "num_tx_ant": 1,
                     "normalize_channel": True},
                 min_jammer_velocity=0.0,
                 max_jammer_velocity=0.0,
                 jammer_mitigation=None,
                 jammer_mitigation_dimensionality=None,
                 return_jammer_signals=False,
                 return_symbols=False,
                 return_decoder_iterations=False):
        super().__init__()
        self._scenario = scenario
        self._domain = domain
        self._perfect_csi = perfect_csi
        self._perfect_jammer_csi = perfect_jammer_csi
        self._num_silent_pilot_symbols = num_silent_pilot_symbols
        self._silent_pilot_symbol_indices = tf.range(self._num_silent_pilot_symbols)
        self._jammer_present = jammer_present
        self._jammer_mitigation = jammer_mitigation
        self._jammer_mitigation_dimensionality = jammer_mitigation_dimensionality
        self._return_jammer_signals = return_jammer_signals
        self._return_symbols = return_symbols
        self._return_decoder_iterations = return_decoder_iterations
        self._jammer_power = tf.cast(jammer_power, tf.complex64)
        #TODO should these kinds of parameters go into e.g. a dict for the channel parameters?
        self._los = los
        self._indoor_probability = indoor_probability
        self._min_ut_velocity = min_ut_velocity
        self._max_ut_velocity = max_ut_velocity
        self._resample_topology = resample_topology
        self._min_jammer_velocity = min_jammer_velocity
        self._max_jammer_velocity = max_jammer_velocity
        self._channel_parameters = channel_parameters

        self._return_jammer_csi = perfect_jammer_csi and jammer_mitigation
        self._estimate_jammer = jammer_mitigation and not perfect_jammer_csi
        

        # Internally set parameters
        self._carrier_frequency = carrier_frequency
        self._fft_size = fft_size
        self._subcarrier_spacing = subcarrier_spacing
        self._num_ofdm_symbols = num_ofdm_symbols
        # TODO shoud I make the default CP-length NR-compliant?
        self._cyclic_prefix_length = cyclic_prefix_length
        self._maximum_delay_spread = maximum_delay_spread
        # self._pilot_ofdm_symbol_indices = [2, 11]
        self._num_bs_ant = num_bs_ant
        self._num_ut = num_ut
        self._num_ut_ant = num_ut_ant
        self._num_bits_per_symbol = num_bits_per_symbol
        self._coderate = coderate
        self._effective_coderate = coderate if coderate is not None else 1.0

        bs_ut_association = np.zeros([1, self._num_ut])
        bs_ut_association[0, :] = 1
        self._rx_tx_association = bs_ut_association
        self._num_tx = self._num_ut
        self._num_streams_per_tx = self._num_ut_ant


        # Setup an OFDM Resource Grid
        # pilot_pattern = OneHotWithSilencePilotPattern(self._num_tx, self._num_streams_per_tx, self._num_ofdm_symbols, self._fft_size, self._num_silent_pilot_symbols)
        pilot_pattern = OneHotPilotPattern(self._num_silent_pilot_symbols, self._num_tx, self._num_streams_per_tx, self._num_ofdm_symbols, self._fft_size)
        pilot_pattern = PilotPatternWithSilence(pilot_pattern, self._silent_pilot_symbol_indices)
        self._rg = ResourceGrid(num_ofdm_symbols=self._num_ofdm_symbols,
                                fft_size=self._fft_size,
                                subcarrier_spacing=self._subcarrier_spacing,
                                num_tx=self._num_tx,
                                num_streams_per_tx=self._num_streams_per_tx,
                                cyclic_prefix_length=self._cyclic_prefix_length,
                                # pilot_pattern="kronecker",
                                # pilot_ofdm_symbol_indices=self._pilot_ofdm_symbol_indices)
                                pilot_pattern=pilot_pattern)

        # Setup StreamManagement
        self._sm = StreamManagement(self._rx_tx_association, self._num_streams_per_tx)

        # instantiate ut-bs-channel
        self._channel_model = self._generate_channel(self._scenario, num_tx=self._num_ut, num_tx_ant=self._num_ut_ant, **self._channel_parameters)

        # Instantiate other building blocks
        self._binary_source = BinarySource()

        self._n = int(self._rg.num_data_symbols*self._num_bits_per_symbol)
        self._k = int(self._n * self._effective_coderate)
        if coderate is not None:
            self._encoder = LDPC5GEncoder(self._k, self._n, num_bits_per_symbol=self._num_bits_per_symbol)
            self._decoder = (
                ReturnIntermediateLDPC5GDecoder(self._encoder, hard_out=False, **decoder_parameters) if self._return_decoder_iterations
                else LDPC5GDecoder(self._encoder, hard_out=False, **decoder_parameters))

        self._mapper = Mapper("qam", self._num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(self._rg)

        # TODO normalize_channel should be simulation parameter (not only here, multiple places)
        self._ofdm_channel = OFDMChannel(self._channel_model, self._rg, add_awgn=True,
                                         normalize_channel=True, return_channel=True)
        if self._domain == "time":
            # TODO only for 802.11n
            self._l_min, self._l_max = time_lag_discrete_time_channel(self._rg.bandwidth, self._maximum_delay_spread)
            # self._l_min, self._l_max = time_lag_discrete_time_channel(self._rg.bandwidth, maximum_delay_spread=2.0e-6)
            self._l_tot = self._l_max - self._l_min + 1
            self._time_channel = ApplyTimeChannel(self._rg.num_time_samples,
                                                  l_tot=self._l_tot,
                                                  add_awgn=True)
            self._modulator = OFDMModulator(self._rg.cyclic_prefix_length)
            self._demodulator = OFDMDemodulator(self._fft_size, self._l_min, self._rg.cyclic_prefix_length)

        self._remove_nulled_subcarriers = RemoveNulledSubcarriers(self._rg)
        self._ls_est = LSChannelEstimator(self._rg, interpolation_type="nn")
        if self._jammer_mitigation == "ian":
            self._lmmse_equ = IAN.IanLMMSEEqualizer(self._rg, self._sm)
        else:
            self._lmmse_equ = LMMSEEqualizer(self._rg, self._sm)
        self._demapper = Demapper("app", "qam", self._num_bits_per_symbol)

        # TODO discuss: each jammer needs its own channel model, which is similar, but not the same as the ut-bs-channel
        # depending on the model, the jammer could change an existing channel, or sometimes has to create a new channel
        # also depending on the model, different parameters have to be changed
        # should the caller be responsible for creating the channel model, or should the jammer be responsible for creating it?
        # the caller could make mistakes, resulting in the model constructing a wrong channel model
        # if the jammer is responsible, how does it know which parameters to change?
        # maybe a jammer exists for each type of channel model? Then knowable, but not very flexible.
        # maybe something with a callable that modifies parameters, then creates a new channel model?
        # we could also say either give me a channel model (then I use it as-is), a channel-factory and paramters (then I create a new channel model), etc.
        # another option: We internally know of all channel types, and have different behavior for each type
        # IStill stay with the layer approach?, as here we only have to change channel creation, otherwise u-t calculations etc. on all levels have to be changed
        # i.e. stream management, possibly classes using ResourceGrid, Topology, etc.
        # maybe using same channel, just generating symbols to send? But then we e.g. cannot be unnormalized when everyone is normalized etc., + all limitations the channel has (only one type of antenna-array etc) 

        # here I just create a new channel model as a caller
        if self._jammer_present:
            self._num_jammer = jammer_parameters["num_tx"] = jammer_parameters.get("num_tx", 1)
            self._num_jammer_ant = jammer_parameters["num_tx_ant"] = jammer_parameters.get("num_tx_ant", 1)
            jammer_parameters["normalize_channel"] = jammer_parameters.get("normalize_channel", True)
            self._jammer_channel_model = self._generate_channel(self._scenario, num_tx=self._num_jammer, num_tx_ant=self._num_jammer_ant, **self._channel_parameters)
            # self._jammer_channel_model = RayleighBlockFading(1, self._num_bs_ant, jammer_parameters["num_tx"], jammer_parameters["num_tx_ant"])
            if(self._domain == "freq"):
                self._jammer = OFDMJammer(self._jammer_channel_model, self._rg, return_channel=self._return_jammer_csi, **jammer_parameters)
            else:
                if self._return_jammer_csi:
                    # return signal in time domain, jammer freq. response in freq. domain
                    return_domain = ("time", "freq")
                else:
                    return_domain = "time"
                self._jammer = TimeDomainOFDMJammer(self._jammer_channel_model,
                                                    self._rg,
                                                    maximum_delay_spread=self._maximum_delay_spread,
                                                    return_channel=self._return_jammer_csi,
                                                    return_domain=return_domain,
                                                    **jammer_parameters)
        
        if self._jammer_mitigation == "pos":
            self._pos = POS.OrthogonalSubspaceProjector(self._jammer_mitigation_dimensionality)

        self._check_settings()
      
    def _new_ut_topology(self, batch_size):
        """Set new user topology"""
        if self._scenario in ["umi", "uma", "rma"]:
            topology = gen_single_sector_topology(batch_size,
                                                  self._num_ut,
                                                  self._scenario,
                                                  min_ut_velocity=self._min_ut_velocity,
                                                  max_ut_velocity=self._max_ut_velocity,
                                                  indoor_probability=self._indoor_probability)
            self._channel_model.set_topology(*topology, los=self._los)

    def _new_jammer_topology(self, batch_size):
        """Set new jammer topology"""
        if self._jammer_present and self._scenario in ["umi", "uma", "rma"]:
            topology = gen_single_sector_topology(batch_size,
                                                  self._num_jammer,
                                                  self._scenario,
                                                  min_ut_velocity=self._min_jammer_velocity,
                                                  max_ut_velocity=self._max_jammer_velocity,
                                                  indoor_probability=self._indoor_probability)
            self._jammer_channel_model.set_topology(*topology, los=self._los)

    def _generate_channel(self, channel_type, **kwargs):
        """Supports UMi, UMa, RMa, Rayleigh, MultiTapRayleigh"""
        channel_type_to_class = {
            "umi": UMi,
            "uma": UMa,
            "rma": RMa,
            "rayleigh": RayleighBlockFading,
            "multitap_rayleigh": MultiTapRayleighBlockFading,
        }
        channel_class = channel_type_to_class[channel_type]

        if channel_type in ["umi", "uma", "rma"]:
            # only configurable parameters:
            num_tx_ant = kwargs.get("num_tx_ant", 1)
            # Configure antenna arrays
            ut_array = AntennaArray(
                                num_rows=1,
                                num_cols=num_tx_ant,
                                polarization="single",
                                polarization_type="V",
                                antenna_pattern="omni",
                                carrier_frequency=self._carrier_frequency)

            bs_array = AntennaArray(num_rows=1,
                                num_cols=int(self._num_bs_ant/2),
                                polarization="dual",
                                polarization_type="cross",
                                antenna_pattern="38.901",
                                carrier_frequency=self._carrier_frequency)

            channel_parameters = {
                "carrier_frequency": self._carrier_frequency,
                "ut_array": ut_array,
                "bs_array": bs_array,
                "direction": "uplink",
                "enable_pathloss": False,
                "enable_shadow_fading": False,
            }
            if self._scenario in ["umi", "uma"]:
                channel_parameters["o2i_model"] = "low"
        elif channel_type in ["rayleigh", "multitap_rayleigh"]:
            channel_parameters = {
                "num_tx": kwargs["num_tx"],
                "num_tx_ant": kwargs["num_tx_ant"],
                "num_rx": 1,
                "num_rx_ant": self._num_bs_ant,
            }
            if channel_type == "multitap_rayleigh":
                channel_parameters["num_paths"] = kwargs.get("num_paths", 3)
        else:
            raise NotImplementedError(f"Channel type {channel_type} not implemented.")

        return channel_class(**channel_parameters)
            

    def _check_settings(self):
        if self._perfect_jammer_csi:
            assert self._jammer_present,\
            "If jammer CSI is perfect (i.e. returned by the jammer), we need a jammer which returns it."
        if self._estimate_jammer:
            assert self._num_silent_pilot_symbols > 0,\
            "We need silent pilots to estimate the jammer CSI."
        assert self._num_silent_pilot_symbols < self._num_ofdm_symbols,\
        "The number of silent pilots must be smaller than the number of OFDM symbols."
        assert self._domain in ["freq", "time"],\
        "domain must be either 'freq' or 'time'"
        assert self._coderate is None or self._coderate <= 1.0 and self._coderate >= 0,\
        "coderate must be in [0, 1] or None"

    def build(self, input_shape):
        if not self._resample_topology:
            self._new_ut_topology(input_shape[0])
            self._new_jammer_topology(input_shape[0])

    # batch size = number of resource grids (=symbols * subcarriers) per stream
    @tf.function(jit_compile=False)
    def call(self, batch_size, ebno_db, training=None):
        # for good statistics, we simulate a new topology for each batch.
        if self._resample_topology:
            self._new_ut_topology(batch_size)
            self._new_jammer_topology(batch_size)
        # no = ebnodb2no(ebno_db, self._num_bits_per_symbol, coderate=self._effective_coderate, resource_grid=None)
        no = ebnodb2no(ebno_db, self._num_bits_per_symbol, coderate=1.0, resource_grid=None)
        b = self._binary_source([batch_size, self._num_tx * self._num_streams_per_tx, self._k])
        # b = self._binary_source([batch_size, self._num_tx * self._num_streams_per_tx * self._rg.num_data_symbols * self._num_bits_per_symbol])
        if self._coderate is not None:
            c = self._encoder(b)
            c = sionna.utils.flatten_last_dims(c, 2)
        else:
            c = b
        x = self._mapper(c)
        # x: [batch_size, num_tx, num_streams_per_tx, num_data_symbols]
        x = tf.reshape(x, [-1, self._num_tx, self._num_streams_per_tx, self._rg.num_data_symbols])
        x_rg = self._rg_mapper(x)
        if self._domain == "freq":
            # y: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
            # h: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
            y, h = self._ofdm_channel([x_rg, no])
        else:
            a, tau = self._channel_model(batch_size, self._rg.num_time_samples+self._l_tot-1, self._rg.bandwidth)
            h_time = cir_to_time_channel(self._rg.bandwidth, a, tau, self._l_min, self._l_max, normalize=True)
            x_time = self._modulator(x_rg)
            y_time = self._time_channel([x_time, h_time, no])
            y = y_time
            if self._perfect_csi:
                # calculate h in frequency domain
                # h = ofdm_frequency_response_from_cir(a, tau, self._rg, normalize=True)
                # TODO: if this is correct, we can use time_channel only (never need cir)
                h = sionna.channel.time_to_ofdm_channel(h_time, self._rg, self._l_min)
            
        if self._jammer_present:
            # at this point, y might be in time or frequency domain.
            if self._return_jammer_csi:
                y, j = self._jammer([y, self._jammer_power])
            else:
                y = self._jammer([y, self._jammer_power])
        # after (potential) jammer, convert signal to freqency domain. Jammer is configured to always return j in freq. domain.
        if self._domain == "time":
            y = self._demodulator(y)
        if self._estimate_jammer:
            jammer_signals = tf.gather(y, self._silent_pilot_symbol_indices, axis=3)
        if self._jammer_mitigation == "pos":
            if self._return_jammer_csi:
                self._pos.set_jammer_frequency_response(j)
            else:
                self._pos.set_jammer_signals(jammer_signals)
            # we transform y before channel estimation to get correct no_eff automically
            # but hence we have to transform h with perfect_csi=True, but not false
            # we could alternatively transform y and h after channel estimation, but then we have to transform no_eff
            y = self._pos(y)
        elif self._jammer_mitigation == "ian":
            if self._return_jammer_csi:
                # attention: this jammer variance might have to have a different shape than the one used for the jammer
                self._lmmse_equ.set_jammer(j, self._jammer_power)
            else:
                # [batch_size, num_rx, num_ofdm_symbols, fft_size, rx_ant, rx_ant]
                jammer_covariance = covariance_estimation_from_signals(jammer_signals, self._num_ofdm_symbols)
                self._lmmse_equ.set_jammer_covariance(jammer_covariance)
        if self._perfect_csi:
            h_hat = self._remove_nulled_subcarriers(h)
            if self._jammer_mitigation == "pos":
                h_hat = self._pos(h_hat)
            err_var = 0.0
        else:
            h_hat, err_var = self._ls_est([y, no])
        x_hat, no_eff = self._lmmse_equ([y, h_hat, err_var, no])
        llr = self._demapper([x_hat, no_eff])
        if self._coderate is not None:
            llr = self._decoder(llr)
            # during training, we want to return intermediate decoding results. During inference, we want to return the final result.
            if self._return_decoder_iterations and not training:
                llr = llr[..., -1]
                
        if self._return_symbols:
            x = tf.reshape(x, [batch_size, -1])
            x_hat = tf.reshape(x_hat, [batch_size, -1])
            result = (x, x_hat)
        else:
            llr = tf.reshape(llr, [batch_size, -1])
            b = tf.reshape(b, [batch_size, -1])
            result = (b, llr)
        if self._return_jammer_signals:
            result += (jammer_signals,)
        return result

class ReturnIntermediateLDPC5GDecoder(LDPC5GDecoder):
    """
    A wrapper around :class:`sionna.fec.ldpc.decoding.LDPC5GDecoder` which returns the llrs after each iteration.
    This can be useful if gradients through a normal LDPC5GDecoder are not good enough.

    Parameters
    ----------
    encoder: LDPC5GEncoder
        An instance of :class:`~sionna.fec.ldpc.encoding.LDPC5GEncoder`
        containing the correct code parameters.

    trainable: bool
        Defaults to False. If True, every outgoing variable node message is
        scaled with a trainable scalar.

    cn_type: str
        A string defaults to '"boxplus-phi"'. One of
        {`"boxplus"`, `"boxplus-phi"`, `"minsum"`} where
        '"boxplus"' implements the single-parity-check APP decoding rule.
        '"boxplus-phi"' implements the numerical more stable version of
        boxplus [Ryan]_.
        '"minsum"' implements the min-approximation of the CN
        update rule [Ryan]_.

    hard_out: bool
        Defaults to True. If True, the decoder provides hard-decided
        codeword bits instead of soft-values.

    track_exit: bool
        Defaults to False. If True, the decoder tracks EXIT characteristics.
        Note that this requires the all-zero CW as input.

    return_infobits: bool
        Defaults to True. If True, only the `k` info bits (soft or
        hard-decided) are returned. Otherwise all `n` positions are
        returned.

    prune_pcm: bool
        Defaults to True. If True, all punctured degree-1 VNs and
        connected check nodes are removed from the decoding graph (see
        [Cammerer]_ for details). Besides numerical differences, this should
        yield the same decoding result but improved the decoding throughput
        and reduces the memory footprint.

    num_iter: int
        Defining the number of decoder iteration (no early stopping used at
        the moment!).

    output_dtype: tf.DType
        Defaults to tf.float32. Defines the output datatype of the layer
        (internal precision remains tf.float32).
    
    Input
    -----
    llrs_ch: [...,n], tf.float32
        2+D tensor containing the channel logits/llr values.

    Output
    ------
    : [..., n, k], tf.float32
        3+D Tensor of same shape up until the last dimension as ``inputs``
        containing, for each decoder iteration, bit-wise soft-estimates
        (or hard-decided bit-values) of all codeword bits.
        The output of the final decoder iteration is located at [:, -1].
        If ``return_infobits`` is True, only the `k`
        information bits are returned.
    """
    def __init__(self, *args, num_iter=10, **kwargs):
        self._num_internal_iter = num_iter
        # TODO take care of hard_out, return_infobits (only for last iteration)
        super().__init__(*args, **{**kwargs, "stateful": True, "num_iter": 1})

    def build(self, input_shape):
        super().build([input_shape, None])
    
    def call(self, llr_ch):
        llr_decs = []
        msg_vn = None
        for i in range(self._num_internal_iter):
            llr_dec, msg_vn = super().call((llr_ch, msg_vn))
            llr_decs.append(llr_dec)
        return tf.stack(llr_decs, axis=-1)


def relative_singular_values(jammer_signals):
    """
    Returns the normalized singular values of the jammer signals.

    Input
    -----
    jammer_signals: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
        received jammer signals during silent pilot symbols
    
    Output
    ------
    : [num_rx * num_rx_ant], float
        Singular values, sorted in descending order. Normalize to sum of eigenvalues = 1.
    """
    j = tf.transpose(jammer_signals, [0, 4, 1, 2, 3])
    # [batch_size, fft_size, num_rx*rx_ant, num_ofdm_symbols]
    j = sionna.utils.flatten_dims(j, 2, 2)
    sigma = tf.linalg.svd(j, compute_uv=False)
    trace = tf.reduce_sum(sigma, axis=-1)
    # normalize to sum of eigenvalues = 1
    sigma = sigma / trace[:, :, None]
    return tf.reduce_mean(sigma, axis=(0, 1))

def singular_values(jammer_signals):
    """
    Returns the unnormalized (and not averaged) singular values of the jammer signals.

    Input
    -----
    jammer_signals: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
        received jammer signals during silent pilot symbols
    
    Output
    ------
    : [batch_size, fft_size, num_rx * num_rx_ant], float
        Singular values, sorted in descending order. Normalize to sum of eigenvalues = 1.
    """
    j = tf.transpose(jammer_signals, [0, 4, 1, 2, 3])
    # [batch_size, fft_size, num_rx*rx_ant, num_ofdm_symbols]
    j = sionna.utils.flatten_dims(j, 2, 2)
    sigma = tf.linalg.svd(j, compute_uv=False)
    return sigma

def bar_plot(values):
    """
    Display a bar plot of the values.

    Input
    -----
    values : list

    """
    plt.bar(np.arange(len(values)), values)
    plt.show()


def simulate_single(ebno_db):
    """
    Simulates a single call to the model and prints the BER.

    Input
    -----
    ebno_db : float
        The SNR (Eb/N0) of the signal sent by the UTs in dB.
    """
    model = Model(**model_parameters)
    b, llr = model(BATCH_SIZE, ebno_db)
    b_hat = sionna.utils.hard_decisions(llr)
    ber = compute_ber(b, b_hat)
    print(ber)

def simulate_single(model, ebno_db):
    b, llr = model(BATCH_SIZE, ebno_db)
    b_hat = sionna.utils.hard_decisions(llr)
    ber = compute_ber(b, b_hat)
    print(ber)

def simulate(legend): 
    model = Model(**model_parameters)
    simulate_model(model, legend)

def simulate_model(model, legend, add_bler=False, verbose=True):
    """
    Simulates the model on SNRs given by ``ebno_dbs`` and plots the BER.

    Input
    -----
    model : :class:`~jammer.model.Model`
        The model to simulate.
    legend : str
        The legend to use in the plot.
    add_bler : bool, optional
        If True, the BLER will be calculated and added to the plot. The default is False.
    """
    ber_plots.simulate(model,
                    ebno_dbs=ebno_dbs,
                    batch_size=BATCH_SIZE,
                    legend=legend,
                    soft_estimates=True,
                    max_mc_iter=MAX_MC_ITER,
                    add_bler=add_bler,
                    show_fig=False,
                    verbose=verbose)
    
def train_model(model,
                learning_rate=0.001,
                loss_fn=None,
                ebno_db=0.0,
                loss_over_logits=None,
                num_iterations=5000,
                weights_filename="weights.pickle",
                log_tensorboard=False,
                validate_ber_tensorboard=False,
                log_weight_images=False,
                show_final_weights=False):
    """
    Trains the model on a single SNR and saves all model weights in a file. Optionally, the loss as well as the weights can be logged to tensorboard.
    
    Input
    -----
    model : :class:`~jammer.simulation_model.Model`
        The model to train.
    loss_fn : callable, optional
        The loss function to use. It should subclass (or implement the interface of) :class:`tf.keras.losses.Loss`.
        Depending on ``model._return_symbols`` and ``loss_over_logits``, the loss function is applied to the symbols, bit-LLRs or the bit estimates.
        If None, an L1-loss is used. The default is None.
    ebno_db : float, optional
        The SNR on which to train the model. The default is 0.0.
    loss_over_logits : bool, optional
        If True, a sigmoid function is applied to the input of the loss function before calculating the loss.
        This is only done if ``model._return_symbols`` is False (otherwise, this parameter is ignored).
        The default is None, which means the input is used as-is.
    num_iterations : int, optional
        The number of batches to train the model. The default is 5000.
    weights_filename : str, optional
        The name of the file to save the weights in. The default is "weights.pickle".
    log_tensorboard : bool, optional
        If True, the loss is logged to tensorboard. The default is False.
    validate_ber_tensorboard : bool, optional
        If True, after training, fast BER-plot is logged to tensorboard as a scalar. The default is False.
    log_weight_images : bool, optional
        If True, the learned weights are logged to tensorboard as images every 500 iterations. The default is False.
    show_final_weights : bool, optional
        If True, the learned weights are plotted after training. The default is False.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # TODO could take average to make it less jittery. Worth it?
    # mean_loss = tf.keras.metrics.Mean(name='train_loss')
    name = weights_filename.split("/")[-1].rsplit(".", 1)[0]
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    if log_tensorboard:
        train_log_dir = 'logs/tensorboard/' + current_time + '-' + name# + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        
    # TODO evaluate if expected bitflips is not just L1 loss. We can then write it as below
    # if loss_fn is None:
    #     if model._return_symbols:
    #         # negative L1 loss
    #         loss_fn = negative_function(MeanAbsoluteError())
    #     else:
    #         loss_fn = negative_function(expected_bitflips)
    #         loss_over_logits = False
    if loss_fn is None:
        loss_fn = negative_function(MeanAbsoluteError())
        if not model._return_symbols:
            loss_over_logits = False
    

    for i in range(num_iterations):
        with tf.GradientTape() as tape:
            label, predicted = model(BATCH_SIZE, ebno_db, training=True)
            if not model._return_symbols and not loss_over_logits:
                predicted = tf.sigmoid(predicted)
            loss = loss_fn(label, predicted)
        # Computing and applying gradients
        weights = model.trainable_weights
        grads = tape.gradient(loss, weights)
        optimizer.apply_gradients(zip(grads, weights))
        # log progress to tensorboard and console
        if i % 10 == 0:
            print(f"{i}/{num_iterations}  Loss: {loss:.2E}", end="\r")
            if log_tensorboard:
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss, step=i)
        if i % 500 == 0 and log_weight_images:
            with train_summary_writer.as_default():
                image = matrix_to_image(model._jammer._weights)
                tf.summary.image("weights", image, step=i)
    
    #BER validation
    if validate_ber_tensorboard:
        validate_log_dir = 'logs/tensorboard/' + current_time + '-' + name# + '/validate'
        tensorboard_validate_model(model, validate_log_dir)

    # Save the weights in a file
    weights = model.get_weights()
    with open(weights_filename, 'wb') as f:
        pickle.dump(weights, f)

    if show_final_weights:
        plot_matrix(model._jammer._weights)
        plt.title(weights_filename)
        plt.show()


def tensorboard_validate_model(model, log_dir):
    """ TODO Simple BER validation on tensorboard"""
    # setup tensorboard
    train_summary_writer = tf.summary.create_file_writer(log_dir)
    # validate
    # TODO we should have a different batch size for validation. Hence, this should be a parameter.
    ber, _ = sionna.utils.sim_ber(model, ebno_dbs, 8, MAX_MC_ITER, soft_estimates=True, verbose=False)
    # ber, _ = sionna.utils.sim_ber(model, ebno_dbs, BATCH_SIZE, MAX_MC_ITER, soft_estimates=True, verbose=False)
    for i, ber_value in enumerate(ber):
        with train_summary_writer.as_default():
            tf.summary.scalar('BER', ber_value, step=i)


def load_weights(model, weights_filename="weights.pickle"):
    """
    Loads the weights from a file into the model.
    
    Input
    -----
    model : :class:`~jammer.model.Model`
        The model to load the weights into.
    weights_filename : str, optional
        The name of the file containing the weights. The default is "weights.pickle".
    """
    # run model once to initialize weights
    # model(BATCH_SIZE, 10.0)
    model(1, 10.0)
    with open(weights_filename, 'rb') as f:
        weights = pickle.load(f)
    model.set_weights(weights)

def mean_L1_loss(model, ebno_db, num_iterations):
    assert model._return_symbols, "Model must return symbols to calculate L1 loss."
    model.return_symbols = True
    mean_loss = tf.keras.metrics.Mean(name='train_loss')
    for i in range(num_iterations):
        x, x_hat = model(BATCH_SIZE, ebno_db)
        # L1 loss
        loss = tf.reduce_mean(tf.reduce_sum(tf.abs(x - x_hat), axis=-1))
        mean_loss.update_state(loss)
    return mean_loss.result()

def negative_function(fn):
    """
        Given a callable with numerical output, returns a callable with the same output, but with the sign flipped.

        Input
        -----
        fn : callable
            A callable with numerical output.
            
        Output
        ------
        : callable
            A callable with the same output as ``fn``, but with the sign flipped.
    """
    def negative_fn(*args, **kwargs):
        return -fn(*args, **kwargs)
    return negative_fn

# TODO perfect_jammer_csi is kind of useless (only used in 2 other variables). Also, jammer csi is not returned unless we need it (i.e. jammer mitigation)

def wifi_vs_5g():
    # TODO make maximum delay spread parameter of model (and jammer?)?
    model_parameters["domain"] = "time"
    model_parameters["jammer_present"] = True
    model_parameters["jammer_power"] = 316.0
    model_parameters["jammer_mitigation"] = "pos"
    model_parameters["num_silent_pilot_symbols"] = 50
    model_parameters["num_ofdm_symbols"] = 64
    model_parameters["return_jammer_signals"] = True
    ebno_db = 30.0
    name = "5G vs. 802.11n"
    # simulate_single(ebno_db)
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    fig.set_size_inches(25, 6)
    axis = {"umi": ax0, "uma": ax1, "rma": ax2}
    for scenario in ["umi", "uma", "rma"]:
        model_parameters["scenario"] = scenario
        for j, protocol in enumerate(["802.11n", "5G"]):
        # for j, protocol in enumerate(["5G"]):
            if protocol == "802.11n":
                model_parameters["carrier_frequency"] = 5.0e9
                model_parameters["fft_size"] = 64
                model_parameters["subcarrier_spacing"] = 312.5e3
                model_parameters["cyclic_prefix_length"] = 16
            else:
                model_parameters["carrier_frequency"] = 3.5e9
                model_parameters["fft_size"] = 128
                model_parameters["subcarrier_spacing"] = 30e3
                model_parameters["cyclic_prefix_length"] = 20
            rel_svs = []
            model = Model(**model_parameters)
            # for i in range(1000):
            for i in range(1000):
                if i % 100 == 0:
                    print(i)
                b, llr, jammer_signals = model(BATCH_SIZE, ebno_db)
                rel_svs.append(relative_singular_values(jammer_signals))
            rel_svs = tf.stack(rel_svs)
            mean = tf.reduce_mean(rel_svs, axis=0)
            std = tf.math.reduce_std(rel_svs, axis=0)
            # plot
            #log scale
            # axis[scenario].set_yscale("log")
            axis[scenario].bar(np.arange(len(mean))+(j-0.5)*0.3, mean, 0.3, yerr=2*std, label=f"{protocol}")
        axis[scenario].set_title(scenario)
        axis[scenario].legend(loc="upper right")
    fig.suptitle(name)
    plt.savefig(f"{name}.png")
    # plt.show()

def indoor_vs_outdoor():
    model_parameters["domain"] = "time"
    model_parameters["jammer_present"] = True
    model_parameters["jammer_power"] = 316.0
    model_parameters["jammer_mitigation"] = "pos"
    model_parameters["num_silent_pilot_symbols"] = 50
    model_parameters["num_ofdm_symbols"] = 64
    model_parameters["return_jammer_signals"] = True
    ebno_db = 30.0
    name = "Indoor vs. Outdoor, LoS vs. NLoS"
    # simulate_single(ebno_db)
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    fig.set_size_inches(25, 6)
    axis = {"umi": ax0, "uma": ax1, "rma": ax2}
    for scenario in ["umi", "uma", "rma"]:
        model_parameters["scenario"] = scenario
        for j, setting in enumerate([("los", "indoor"), ("los", "outdoor"), ("nlos", "indoor"), ("nlos", "outdoor")]):
            los, indoor = setting
            model_parameters["los"] = True if los == "los" else False
            model_parameters["indoor_probability"] = 1.0 if indoor == "indoor" else 0.0
            rel_svs = []
            model = Model(**model_parameters)
            for i in range(1000):
                if i % 100 == 0:
                    print(i)
                b, llr, jammer_signals = model(BATCH_SIZE, ebno_db)
                rel_svs.append(relative_singular_values(jammer_signals))
            rel_svs = tf.stack(rel_svs)
            mean = tf.reduce_mean(rel_svs, axis=0)
            std = tf.math.reduce_std(rel_svs, axis=0)
            # plot
            #log scale
            # axis[scenario].set_yscale("log")
            axis[scenario].bar(np.arange(len(mean))+(j-2)*0.15, mean, 0.15, yerr=2*std, label=f"{los}, {indoor}")
        axis[scenario].set_title(scenario)
        axis[scenario].legend(loc="upper right")
    fig.suptitle(name)
    plt.savefig(f"{name}.png", bbox_inches='tight')
    # plt.show()

def multi_jammers():
    model_parameters["domain"] = "time"
    model_parameters["jammer_present"] = True
    model_parameters["jammer_power"] = 316.0
    model_parameters["jammer_mitigation"] = "pos"
    model_parameters["num_silent_pilot_symbols"] = 50
    model_parameters["num_ofdm_symbols"] = 64
    model_parameters["return_jammer_signals"] = True
    ebno_db = 30.0
    name = "Multiple Jammers"
    # simulate_single(ebno_db)
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    fig.set_size_inches(25, 6)
    axis = {"umi": ax0, "uma": ax1, "rma": ax2}
    for scenario in ["umi", "uma", "rma"]:
        for j, jammer_antennas in enumerate([1, 2, 4]):
            rel_svs = []
            jammer_parameters["num_tx"] = jammer_antennas
            model_parameters["scenario"] = scenario
            model = Model(**model_parameters)
            for i in range(1000):
            # for i in range(10):
                if i % 100 == 0:
                    print(i)
                b, llr, jammer_signals = model(BATCH_SIZE, ebno_db)
                rel_svs.append(relative_singular_values(jammer_signals))
            rel_svs = tf.stack(rel_svs)
            mean = tf.reduce_mean(rel_svs, axis=0)
            std = tf.math.reduce_std(rel_svs, axis=0)
            # plot
            #log scale
            # axis[scenario].set_yscale("log")
            axis[scenario].bar(np.arange(len(mean))+(j-1)*0.2, mean, 0.2, yerr=2*std, label=f"{jammer_antennas} jammer(s)")
        axis[scenario].set_title(scenario)
        axis[scenario].legend(loc="upper right")
    fig.suptitle(name)
    # plt.savefig(f"{name}.png")
    plt.show()

#training
# model_parameters["perfect_csi"] = False
# model_parameters["jammer_present"] = True
# model_parameters["jammer_mitigation"] = "pos"
# model_parameters["jammer_mitigation_dimensionality"] = 1
# jammer_parameters["trainable"] = True
# model_parameters["return_symbols"] = True

# # jammer which sends during jammer-pilots, but is able to learn during rest
# filename = "datalearning_weights_4ue.pickle"
# model_parameters["num_ut"] = 1
# jammer_parameters["trainable_mask"] = tf.concat([tf.zeros([4,128], dtype=bool), tf.ones([10,128], dtype=tf.bool)], axis=0)
# model_train = Model(**model_parameters)
# train_model(model_train, 5000, filename, log_tensorboard=True, log_weight_images=True)

# jammer which can choose any rg-element to send on
# filename = "whole_rg_weights_4ue_pow1.pickle"
# model_parameters["num_ut"] = 4
# filename = "whole_rg_weights_1ue_pow1.pickle"
# model_parameters["num_ut"] = 1
# model_parameters["jammer_power"] = 1.0
# jammer_parameters["trainable_mask"] = tf.ones([14,128], dtype=bool)
# model_train = Model(**model_parameters)
# train_model(model_train, weights_filename=filename, log_tensorboard=True, log_weight_images=True)

# ber_plots.title = "Learning Jammers, PoS Mitigation"
# new_cycler = plt.cycler('linestyle', ['--', '-', '--', '-']) + plt.cycler('color', ['blue', 'blue', 'orange', 'orange'])
# plt.rcParams['axes.prop_cycle'] = new_cycler
# BATCH_SIZE = 16
# MAX_MC_ITER = 50
# NUM_SNR_POINTS = 10
# ebno_dbs = np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, NUM_SNR_POINTS)

# jammer_parameters["trainable"] = False
# model_parameters["return_symbols"] = False
# power = 1.0
# # untrained jammer, only over non-silent symbols
# for num_ut in [1, 4]:
#     model_parameters["num_ut"] = num_ut
#     model_parameters["jammer_power"] = normalize_power(tf.concat([tf.zeros([4,1]), tf.ones([10,1])], axis=0), is_amplitude=False) * power
#     model = Model(**model_parameters)
#     simulate_model(model, f"Untrained Jammer, UEs: {num_ut}")
# # trained jammer
# for num_ut in [1, 4]:
#     filename = f"whole_rg_weights_{num_ut}ue_pow1.pickle"
#     model_parameters["num_ut"] = num_ut
#     model_parameters["jammer_power"] = power
#     jammer_parameters["trainable_mask"] = tf.ones([14,128], dtype=bool)
#     model = Model(**model_parameters)
#     load_weights(model, filename)
#     simulate_model(model, f"Trained Jammer, UEs: {num_ut}")
# ber_plots(ylim=[1.0e-2, 1.0])
    

# # sanity checks
# ber_plots.title = "Sanity Checks"
# power = 1.0
# jammer_parameters["trainable"] = False
# model_parameters["return_symbols"] = False
# model_parameters["num_ut"] = 1

# model_parameters["jammer_power"] = normalize_power(tf.concat([tf.zeros([4,1]), tf.ones([10,1])], axis=0), is_amplitude=False) * power
# model = Model(**model_parameters)
# simulate_model(model, "uniform nonsilent-symbol jammer")

# model_parameters["jammer_power"] = normalize_power(tf.concat([tf.zeros([4,1]), tf.ones([1,1]), tf.zeros([9,1])], axis=0), is_amplitude=False) * power
# model = Model(**model_parameters)
# simulate_model(model, "pilot only jammer")

# model_parameters["jammer_power"] = power
# jammer_parameters["trainable_mask"] = tf.ones([14,128], dtype=bool)
# jammer_parameters["trainable"] = False
# model = Model(**model_parameters)
# load_weights(model, "whole_rg_weights_1ue_pow1.pickle")
# simulate_model(model, "trained jammer")

# ber_plots(ylim=[1.0e-2, 1.0])


# filename = "symbol_weights.pickle"
# jammer_parameters["trainable_mask"] = tf.ones([14, 1], dtype=bool)
# model_train = Model(**model_parameters)
# train_model(model_train, 5000, filename, log_tensorboard=True, log_weight_images=True)

# # jammer which can choose non-silent symbol times
# filename = "nonsilent_symbol_weights.pickle"
# jammer_parameters["trainable_mask"] = tf.concat([tf.zeros([4, 1], dtype=bool), tf.ones([10, 1], dtype=bool)], axis=0)
# model_train = Model(**model_parameters)
# train_model(model_train, 20000, filename, log_tensorboard=True, log_weight_images=True)

# # inference
# jammer_parameters["trainable"] = False
# simulate("Untrained Jammer")
# # for filename, trainable_mask in [("whole_rg_weights.pickle", tf.ones([14,128], dtype=bool)), ("symbol_weights.pickle", tf.ones([14, 1], dtype=bool))]:
# for filename, trainable_mask in [("whole_rg_weights.pickle", tf.concat([tf.zeros([4,128], dtype=bool), tf.ones([10,128], dtype=tf.bool)])]:
#     jammer_parameters["trainable_mask"] = trainable_mask
#     model = Model(**model_parameters)
#     load_weights(model, filename)
#     simulate_model(model, filename)
#     # calculate BCE
#     mean_bce = tf.keras.metrics.Mean(name='bce')
#     mean_bitflips = tf.keras.metrics.Mean(name='bitflips')
#     for i in range(10):
#         b, llr = model(BATCH_SIZE, 10.0)
#         bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#         mean_bce.update_state(bce(b, llr))
#         bitflips = utils.expected_bitflips(b, tf.sigmoid(llr))
#         mean_bitflips.update_state(bitflips)
#     print(f"{filename}: BCE: {mean_bce.result()}, Bitflips: {mean_bitflips.result()}")
# ber_plots()



# model_parameters["num_ut"] = 1
# model_parameters["perfect_csi"] = True
# model_parameters["num_silent_pilot_symbols"] = 8
# model_parameters["domain"] = "time"
# simulate("No Jammer")
# model_parameters["jammer_present"] = True
# model_parameters["jammer_power"] = db_to_linear(25.0)
# simulate("Jammer")

# Time-Domain Mitigation Strategy evaluation
# BATCH_SIZE = 2
# experiment_max_mc_iter = 5000

# # jammer_power_db = 10.0
# jammer_power_db = 25.0
# perfect_csi = True
# MAX_MC_ITER = experiment_max_mc_iter
# model = Model(num_ut=1, perfect_csi=perfect_csi, num_silent_pilot_symbols=8, domain="time")
# simulate_model(model, "No Jammer")
# for i in [0, 1, 4]:
#     model = Model(num_ut=1, perfect_csi=perfect_csi, num_silent_pilot_symbols=8, domain="time",
#                   jammer_present=True, jammer_power=db_to_linear(jammer_power_db), jammer_mitigation="pos", perfect_jammer_csi=False,
#                   jammer_mitigation_dimensionality=i)
#     simulate_model(model, f"POS, {i} dimensions")
# for coderate in [0.5, 0.25]:
#     MAX_MC_ITER = experiment_max_mc_iter / coderate
#     model = Model(num_ut=1, perfect_csi=perfect_csi, num_silent_pilot_symbols=8, domain="time",
#                   jammer_present=True, jammer_power=db_to_linear(jammer_power_db), jammer_mitigation="pos", perfect_jammer_csi=False,
#                   jammer_mitigation_dimensionality=1, coderate=coderate)
#     simulate_model(model, f"POS 1 dimension, {coderate} coderate")
# MAX_MC_ITER = experiment_max_mc_iter
# model = Model(num_ut=1, perfect_csi=perfect_csi, num_silent_pilot_symbols=8, domain="time",
#               jammer_present=True, jammer_power=db_to_linear(jammer_power_db), jammer_mitigation="pos", perfect_jammer_csi=False,
#               jammer_mitigation_dimensionality=1, coderate=None,
#               jammer_parameters={"send_cyclic_prefix": True,
#                                  "num_tx_ant": 1,
#                                  "num_tx": 1,
#                                  "normalize_channel": True})
# simulate_model(model, "Jammer with CP, POS, 1 dimension")
# ber_plots.title = f"1 Time-Domain Jammer ({int(jammer_power_db)}dB) without CP, estimated Jammer CSI."
# ber_plots(ylim=[1.0e-5, 1.0], save_fig=True, path=f"time_domain_mitigation_{int(jammer_power_db)}db_bak.png")
# ber_plots.reset()

# model_parameters["domain"] = "time"
# model_parameters["jammer_present"] = True
# model_parameters["jammer_power"] = 316.0
# model_parameters["jammer_mitigation"] = "pos"
# model_parameters["num_silent_pilot_symbols"] = 50
# model_parameters["num_ofdm_symbols"] = 64
# simulate("Time Domain, POS")

# model_parameters["coderate"] = 0.5
# model_parameters["domain"] = "time"
# simulate("Time Domain, no jammer")
# model_parameters["jammer_present"] = True
# model_parameters["perfect_jammer_csi"] = True
# simulate("Time Domain, Jammer with CP")
# model_parameters["domain"] = "freq"
# simulate("Freq. Domain, Jammer with CP")
# model_parameters["perfect_jammer_csi"] = False
# model_parameters["jammer_present"] = False
# simulate("Freq. Domain, no jammer")

# model_parameters["scenario"] = "multitap_rayleigh"
# ber_plots.title = "Time Domain. Perfect CSI. POS."
# model_parameters["jammer_mitigation"] = "ian"
# model_parameters["perfect_jammer_csi"] = True
# model_parameters["jammer_present"] = True
# model_parameters["jammer_power"] = 250
# model_parameters["domain"] = "freq"
# simulate("Freq. Domain")
# model_parameters["domain"] = "time"
# jammer_parameters["send_cyclic_prefix"] = True
# simulate("Time Domain, Jammer with CP")
# jammer_parameters["send_cyclic_prefix"] = False
# simulate("Time Domain, Jammer without CP")

# jammer_parameters["jamming_type"] = "pilot"
# model_parameters["jammer_present"] = True
# model_parameters["jammer_mitigation"] = "pos"
# simulate("LMMSE with Jammer, POS")

# model_parameters["jammer_present"] = True
# model_parameters["jammer_mitigation"] = "ian"
# simulate("LMMSE with Jammer, IAN")

# for jammer_mitigation in ["pos", "ian"]:
#     ber_plots.reset()
#     model_parameters["perfect_csi"] = False
#     model_parameters["jammer_present"] = True
#     model_parameters["perfect_jammer_csi"] = True
#     model_parameters["jammer_mitigation"] = jammer_mitigation
#     simulate("Perfect Jammer CSI")

#     model_parameters["perfect_jammer_csi"] = False
#     for num_silent_pilot_symbols in range(1, 14, 3):
#         model_parameters["num_silent_pilot_symbols"] = num_silent_pilot_symbols
#         simulate(f"estimated CSI, {num_silent_pilot_symbols} silent pilots")
#     ber_plots.title = f"Jammer Mitigation, {jammer_mitigation.upper()}"
#     ber_plots()


# # simulate jammers with different samplers
# for sampler in [sionna.mapping.Constellation("qam", 2), "gaussian", "uniform", lambda shape, dtype: tf.ones(shape, dtype=dtype)]:
#     jammer_parameters["sampler"] = sampler
#     simulate(f"LMMSE with Jammer, {sampler}")


# # simulate jammed models with different parameters (strength and number of antennas)
# model_parameters["jammer_present"] = True
# for num_tx_ant in range(5, 1, -1):
#   jammer_parameters["num_tx_ant"] = num_tx_ant
#   simulate(f"LMMSE with Jammer, {num_tx_ant} antennas")

# jammer_parameters["num_tx_ant"] = 1
# for jammer_power in np.arange(1.0, 0.0, -0.2):
#     model_parameters["jammer_power"] = jammer_power
#     simulate(f"LMMSE with Jammer, {jammer_power} power")

# # adjust colors
# next_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][5]
# # cycler taking last color, but changing alpha
# from cycler import cycler, concat
# already_used_cycler = cycler(color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0:6]) * cycler(alpha=[1.0])
# new_cycler = (cycler(color=[next_color]) * cycler(alpha=[1.0, 0.8, 0.6, 0.4, 0.2]))
# new_cycler = concat(already_used_cycler, new_cycler)
# plt.rcParams['axes.prop_cycle'] = new_cycler




# ber_plots()

# %%

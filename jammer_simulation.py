#%%
import os
import drjit
# gpu_num = 2 # Use "" to use the CPU
gpu_num = [0, 1, 2, 3] # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sionna
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
tf.get_logger().setLevel('ERROR')
# tf.config.run_functions_eagerly(True)

import matplotlib.pyplot as plt
import numpy as np
import time
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

from jammer.jammer import OFDMJammer, TimeDomainOFDMJammer
from jammer.mitigation import POS, IAN
from custom_pilots import OneHotWithSilencePilotPattern, OneHotPilotPattern, PilotPatternWithSilence
from channel_models import MultiTapRayleighBlockFading
from jammer.utils import covariance_estimation_from_signals, ofdm_frequency_response_from_cir


# sionna.config.xla_compat=True
class Model(tf.keras.Model):
    """Simulate OFDM MIMO transmissions over a 3GPP 38.901 model. No coding for now.
    """
    def __init__(self, scenario, domain="freq", los=None, indoor_probability=0.8, perfect_csi=False, perfect_jammer_csi=False, num_silent_pilot_symbols=0, jammer_present=False, jammer_power=1.0, jammer_parameters={}, jammer_mitigation=None, return_jammer_signals=False):
        super().__init__()
        self._scenario = scenario
        self._domain = domain
        self._perfect_csi = perfect_csi
        self._perfect_jammer_csi = perfect_jammer_csi
        self._num_silent_pilot_symbols = num_silent_pilot_symbols
        self._silent_pilot_symbol_indices = tf.range(self._num_silent_pilot_symbols)
        self._jammer_present = jammer_present
        self._jammer_mitigation = jammer_mitigation
        self._return_jammer_signals = return_jammer_signals
        self._jammer_power = tf.cast(jammer_power, tf.complex64)
        #TODO should these kinds of parameters go into e.g. a dict for the channel parameters?
        self._los = los
        self._indoor_probability = indoor_probability

        self._return_jammer_csi = perfect_jammer_csi and jammer_mitigation
        self._estimate_jammer_covariance = jammer_mitigation in ["pos", "ian"] and not perfect_jammer_csi
        

        # Internally set parameters
        self._carrier_frequency = 3.5e9
        self._fft_size = 128
        self._subcarrier_spacing = 30e3
        # self._num_ofdm_symbols = 14
        self._num_ofdm_symbols = 64
        self._cyclic_prefix_length = 20
        # self._pilot_ofdm_symbol_indices = [2, 11]
        self._num_bs_ant = 16
        self._num_ut = 4
        self._num_ut_ant = 1
        self._num_bits_per_symbol = 2

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
        self._channel_model = self._generate_channel(self._scenario, num_tx=self._num_ut, num_tx_ant=self._num_ut_ant)

        # Instantiate other building blocks
        self._binary_source = BinarySource()
        self._qam_source = QAMSource(self._num_bits_per_symbol)

        self._mapper = Mapper("qam", self._num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(self._rg)

        # TODO normalize_channel should be simulation parameter (not only here, multiple places)
        self._ofdm_channel = OFDMChannel(self._channel_model, self._rg, add_awgn=True,
                                         normalize_channel=True, return_channel=True)
        if self._domain == "time":
            self._l_min, self._l_max = time_lag_discrete_time_channel(self._rg.bandwidth)
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
            self._num_jammer = jammer_parameters["num_tx"]
            self._num_jammer_ant = jammer_parameters["num_tx_ant"]
            self._jammer_channel_model = self._generate_channel(self._scenario, num_tx=self._num_jammer, num_tx_ant=self._num_jammer_ant)
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
                                                    self._rg, return_channel=self._return_jammer_csi,
                                                    return_domain=return_domain,
                                                    **jammer_parameters)
        
        if self._jammer_mitigation == "pos":
            self._pos = POS.OrthogonalSubspaceProjector()
        
        self._check_settings()
        
    def new_ut_topology(self, batch_size):
        """Set new user topology"""
        if self._scenario in ["umi", "uma", "rma"]:
            topology = gen_single_sector_topology(batch_size,
                                                  self._num_ut,
                                                  self._scenario,
                                                  min_ut_velocity=0.0,
                                                  max_ut_velocity=0.0)
            self._channel_model.set_topology(*topology)

    def new_jammer_topology(self, batch_size):
        """Set new jammer topology"""
        if self._jammer_present and self._scenario in ["umi", "uma", "rma"]:
            topology = gen_single_sector_topology(batch_size,
                                                  self._num_jammer,
                                                  self._scenario,
                                                  min_ut_velocity=0.0,
                                                  max_ut_velocity=0.0)
            self._jammer_channel_model.set_topology(*topology)

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
        if not self._perfect_jammer_csi:
            assert self._num_silent_pilot_symbols > 0,\
            "If jammer csi is not perfect, we need silent pilots to estimate the jammer CSI."
        assert self._num_silent_pilot_symbols < self._num_ofdm_symbols,\
        "The number of silent pilots must be smaller than the number of OFDM symbols."
        assert self._domain in ["freq", "time"],\
        "domain must be either 'freq' or 'time'"

    # batch size = number of resource grids (=symbols * subcarriers) per stream
    @tf.function(jit_compile=False)
    def call(self, batch_size, ebno_db):
        # for good statistics, we simulate a new topology for each batch.
        self.new_ut_topology(batch_size)
        self.new_jammer_topology(batch_size)
        # no = ebnodb2no(ebno_db, self._num_bits_per_symbol, coderate=1.0, resource_grid=self._rg)
        no = ebnodb2no(ebno_db, self._num_bits_per_symbol, coderate=1.0, resource_grid=None)
        b = self._binary_source([batch_size, self._num_tx * self._num_streams_per_tx * self._rg.num_data_symbols * self._num_bits_per_symbol])
        x = self._mapper(b)
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
        if self._estimate_jammer_covariance:
            # TODO: one of the next 2 lines is slow. Benchmark and optimize. Might be tf.gather. Should we only allow connected slices?
            jammer_signals = tf.gather(y, self._silent_pilot_symbol_indices, axis=3)
            # # code to display jammer dimensionality
            # bar_plot(relative_singular_values(jammer_signals).numpy())
            # [batch_size, num_rx, num_ofdm_symbols, fft_size, rx_ant, rx_ant]
            jammer_covariance = covariance_estimation_from_signals(jammer_signals, self._num_ofdm_symbols)
        if self._jammer_mitigation == "pos":
            if self._return_jammer_csi:
                self._pos.set_jammer(j)
            else:
                self._pos.set_jammer_covariance(jammer_covariance)
            # we transform y before channel estimation to get correct no_eff automically
            # but hence we have to transform h with perfect_csi=True, but not false
            # we could alternatively transform y and h after channel estimation, but then we have to transform no_eff
            y = self._pos(y)
        elif self._jammer_mitigation == "ian":
            if self._return_jammer_csi:
                # attention: this jammer variance might have to have a different shape than the one used for the jammer
                self._lmmse_equ.set_jammer(j, self._jammer_power)
            else:
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
        llr = tf.reshape(llr, [batch_size, -1])
        if self._return_jammer_signals:
            return b, llr, jammer_signals
        else:
            return b, llr


def relative_singular_values(jammer_signals):
    """Input: jammer_signals: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]"""
    j = tf.transpose(jammer_signals, [0, 4, 1, 2, 3])
    # [batch_size, fft_size, num_rx*rx_ant, num_ofdm_symbols]
    j = sionna.utils.flatten_dims(j, 2, 2)
    sigma = tf.linalg.svd(j, compute_uv=False)
    trace = tf.reduce_sum(sigma, axis=-1)
    # normalize to sum of eigenvalues = 1
    sigma = sigma / trace[:, :, None]
    return tf.reduce_mean(sigma, axis=(0, 1))

def bar_plot(values):
    plt.bar(np.arange(len(values)), values)
    plt.show()


BATCH_SIZE = 1
MAX_MC_ITER = 30
EBN0_DB_MIN = -5.0
EBN0_DB_MAX = 15.0
NUM_SNR_POINTS = 10
ebno_dbs = np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, NUM_SNR_POINTS)
ber_plots = PlotBER("POS with CSI Estimation")

def simulate_single(ebno_db):
    model = Model(**model_parameters)
    b, llr = model(BATCH_SIZE, ebno_db)
    b_hat = sionna.utils.hard_decisions(llr)
    ber = compute_ber(b, b_hat)
    print(ber)

def simulate(legend): 
    model = Model(**model_parameters)
    ber_plots.simulate(model,
                    ebno_dbs=ebno_dbs,
                    batch_size=BATCH_SIZE,
                    legend=legend,
                    soft_estimates=True,
                    max_mc_iter=MAX_MC_ITER,
                    show_fig=False)

jammer_parameters = {
    "num_tx": 1,
    "num_tx_ant": 1,
    "normalize_channel": True,
}

model_parameters = {
    "scenario": "umi",
    "perfect_csi": True,
    "domain": "freq",
    "los": None,
    "indoor_probability": 0.8,
    "num_silent_pilot_symbols": 4,
    "jammer_present": False,
    "perfect_jammer_csi": False,
    "jammer_mitigation": None,
    "jammer_power": 1.0,
    "return_jammer_signals": False,
    "jammer_parameters": jammer_parameters,
}

# TODO perfect_jammer_csi is kind of useless (only used in 2 other variables). Also, jammer csi is not returned unless we need it (i.e. jammer mitigation)

model_parameters["domain"] = "time"
model_parameters["jammer_present"] = True
model_parameters["jammer_power"] = 316.0
model_parameters["jammer_mitigation"] = "pos"
model_parameters["num_silent_pilot_symbols"] = 50
model_parameters["return_jammer_signals"] = True
ebno_db = 30.0
name = "Log-Scale"
# simulate_single(ebno_db)
fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
fig.set_size_inches(18.5, 6)
axis = {"umi": ax0, "uma": ax1, "rma": ax2}
for scenario in ["umi", "uma", "rma"]:
    rel_svs = []
    model_parameters["scenario"] = scenario
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
    axis[scenario].set_title(scenario)
    #log scale
    # axis[scenario].set_yscale("log")
    axis[scenario].bar(np.arange(len(mean)), mean, yerr=2*std)
fig.suptitle(name)
plt.show()


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

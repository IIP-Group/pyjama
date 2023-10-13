#%%
import os
gpu_num = 5 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sionna
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
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
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel
from sionna.channel import ApplyOFDMChannel, ApplyTimeChannel, OFDMChannel, RayleighBlockFading

from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder

from sionna.mapping import Mapper, Demapper

from sionna.utils import BinarySource, ebnodb2no, sim_ber, plot_ber, QAMSource, PlotBER
from sionna.utils.metrics import compute_ber

# from jammer import OFDMJammer
from jammer.jammer import OFDMJammer
from jammer.mitigation import POS, IAN


# sionna.config.xla_compat=True
class Model(tf.keras.Model):
    """Simulate OFDM MIMO transmissions over a 3GPP 38.901 model. No coding for now.
    """
    def __init__(self, scenario, jammer_present=False, jammer_power=1.0, jammer_parameters={}, jammer_mitigation=None, perfect_csi=False):
        super().__init__()
        self._scenario = scenario
        self._channel_class = {"umi": UMi, "uma": UMa, "rma": RMa}[scenario]
        self._perfect_csi = perfect_csi
        self._jammer_present = jammer_present
        self._jammer_mitigation = jammer_mitigation
        self._jammer_power = jammer_power

        # Internally set parameters
        self._carrier_frequency = 3.5e9
        self._fft_size = 128
        self._subcarrier_spacing = 30e3
        self._num_ofdm_symbols = 14
        self._cyclic_prefix_length = 20
        self._pilot_ofdm_symbol_indices = [2, 11]
        self._num_bs_ant = 8
        self._num_ut = 4
        self._num_ut_ant = 1
        self._num_bits_per_symbol = 2

        bs_ut_association = np.zeros([1, self._num_ut])
        bs_ut_association[0, :] = 1
        self._rx_tx_association = bs_ut_association
        self._num_tx = self._num_ut
        self._num_streams_per_tx = self._num_ut_ant


        # Setup an OFDM Resource Grid
        self._rg = ResourceGrid(num_ofdm_symbols=self._num_ofdm_symbols,
                                fft_size=self._fft_size,
                                subcarrier_spacing=self._subcarrier_spacing,
                                num_tx=self._num_tx,
                                num_streams_per_tx=self._num_streams_per_tx,
                                cyclic_prefix_length=self._cyclic_prefix_length,
                                pilot_pattern="kronecker",
                                pilot_ofdm_symbol_indices=self._pilot_ofdm_symbol_indices)

        # Setup StreamManagement
        self._sm = StreamManagement(self._rx_tx_association, self._num_streams_per_tx)

        # Configure antenna arrays
        self._ut_array = AntennaArray(
                                 num_rows=1,
                                 num_cols=1,
                                 polarization="single",
                                 polarization_type="V",
                                 antenna_pattern="omni",
                                 carrier_frequency=self._carrier_frequency)

        self._bs_array = AntennaArray(num_rows=1,
                                      num_cols=int(self._num_bs_ant/2),
                                      polarization="dual",
                                      polarization_type="cross",
                                      antenna_pattern="38.901",
                                      carrier_frequency=self._carrier_frequency)

        channel_parameters = {
            "carrier_frequency": self._carrier_frequency,
            "ut_array": self._ut_array,
            "bs_array": self._bs_array,
            "direction": "uplink",
            "enable_pathloss": False,
            "enable_shadow_fading": False,
        }
        if self._scenario in ["umi", "uma"]:
            channel_parameters["o2i_model"] = "low"
        
        self._channel_model = self._channel_class(**channel_parameters)
        # if self._scenario == "umi":
        #     self._channel_model = UMi(**channel_parameters)
        # elif self._scenario == "uma":
        #     self._channel_model = UMa(**channel_parameters)
        # elif self._scenario == "rma":
        #     self._channel_model = RMa(**channel_parameters)

        # Instantiate other building blocks
        self._binary_source = BinarySource()
        self._qam_source = QAMSource(self._num_bits_per_symbol)

        self._mapper = Mapper("qam", self._num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(self._rg)

        self._ofdm_channel = OFDMChannel(self._channel_model, self._rg, add_awgn=True,
                                         normalize_channel=True, return_channel=True)

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
            jammer_channel_parameters = channel_parameters.copy()
            self._jammer_array = AntennaArray(num_rows=1,
                                        num_cols=jammer_parameters["num_tx_ant"],
                                        polarization="single",
                                        polarization_type="V",
                                        antenna_pattern="omni",
                                        carrier_frequency=self._carrier_frequency)
            jammer_channel_parameters["ut_array"] = self._jammer_array
            self._jammer_channel_model = self._channel_class(**jammer_channel_parameters)
            self._num_jammer = jammer_parameters["num_tx"]
            self._num_jammer_ant = jammer_parameters["num_tx_ant"]
            # self._jammer_channel_model = RayleighBlockFading(1, self._num_bs_ant, jammer_parameters["num_tx"], jammer_parameters["num_tx_ant"])
            self._jammer = OFDMJammer(self._jammer_channel_model, self._rg, return_channel=self._perfect_csi and self._jammer_mitigation, **jammer_parameters)
        
        if self._jammer_mitigation == "pos":
            self._pos = POS.OrthogonalSubspaceProjector()
        
        
    def new_ut_topology(self, batch_size):
        """Set new user topology"""
        topology = gen_single_sector_topology(batch_size,
                                              self._num_ut,
                                              self._scenario,
                                              min_ut_velocity=0.0,
                                              max_ut_velocity=0.0)
        self._channel_model.set_topology(*topology)

    def new_jammer_topology(self, batch_size):
        """Set new jammer topology"""
        if self._jammer_present:
            topology = gen_single_sector_topology(batch_size,
                                                  self._num_jammer,
                                                  self._scenario,
                                                  min_ut_velocity=0.0,
                                                  max_ut_velocity=0.0)
            self._jammer_channel_model.set_topology(*topology)

    def jammer_variance(self, batch_size, dtype=tf.complex64):
        """Overwrite to change variance of jammer signal before channel.
        Output: rho: [batch_size, num_jammer, num_jammer_ant, num_ofdm_symbols, fft_size]
        """
        shape = [batch_size, self._num_jammer, self._num_jammer_ant, self._num_ofdm_symbols, self._fft_size]
        return self._jammer_power * tf.ones(shape, dtype=tf.as_dtype(dtype))

    # batch size = number of resource grids (=symbols * subcarriers) per stream
    
    @tf.function(jit_compile=False)
    def call(self, batch_size, ebno_db):
        # for good statistics, we simulate a new topology for each batch.
        self.new_ut_topology(batch_size)
        self.new_jammer_topology(batch_size)
        no = ebnodb2no(ebno_db, self._num_bits_per_symbol, coderate=1.0, resource_grid=self._rg)
        b = self._binary_source([batch_size, self._num_tx * self._num_streams_per_tx * self._rg.num_data_symbols * self._num_bits_per_symbol])
        x = self._mapper(b)
        # [batch_size, num_tx, num_streams_per_tx, num_data_symbols]
        x = tf.reshape(x, [-1, self._num_tx, self._num_streams_per_tx, self._rg.num_data_symbols])
        x_rg = self._rg_mapper(x)
        y, h = self._ofdm_channel([x_rg, no])
        if self._jammer_present:
            jammer_variance = self.jammer_variance(batch_size, dtype=y.dtype)
            if self._jammer_mitigation and self._perfect_csi:
                y, j = self._jammer([y, jammer_variance])
            else:
                y = self._jammer([y, jammer_variance])
                # TODO restructure (if/else more logical, j estimation)
                # TODO estimate jammer channel
            if self._jammer_mitigation == "pos":
                self._pos.set_jammer(j)
                # we transform y before channel estimation to get correct no_eff automically
                # but hence we have to transform h with perfect_csi=True, but not false
                # we could alternatively transform y and h after channel estimation, but then we have to transform no_eff
                y = self._pos(y)
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
        return b, llr





BATCH_SIZE = 32
EBN0_DB_MIN = -5.0
EBN0_DB_MAX = 15.0
NUM_SNR_POINTS = 10
ebno_dbs = np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, NUM_SNR_POINTS)
ber_plots = PlotBER("")

def simulate(legend): 
    model = Model(**model_parameters)
    ber_plots.simulate(model,
                    ebno_dbs=ebno_dbs,
                    batch_size=BATCH_SIZE,
                    legend=legend,
                    soft_estimates=True,
                    max_mc_iter=20,
                    show_fig=False)

jammer_parameters = {
    "num_tx": 1,
    "num_tx_ant": 2,
    "normalize_channel": True,
}

model_parameters = {
    "scenario": "umi",
    "perfect_csi": True,
    "jammer_present": False,
    "jammer_mitigation": None,
    "jammer_power": 1.0,
    "jammer_parameters": jammer_parameters,
}

simulate("LMMSE without Jammer")

model_parameters["jammer_present"] = True
simulate("LMMSE with Jammer")

model_parameters["jammer_present"] = True
model_parameters["jammer_mitigation"] = "pos"
simulate("LMMSE with Jammer, POS")

model_parameters["jammer_present"] = True
model_parameters["jammer_mitigation"] = "ian"
simulate("LMMSE with Jammer, IAN")


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




ber_plots()

# %%

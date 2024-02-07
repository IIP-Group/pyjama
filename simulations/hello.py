#%%
# Jammer simulation over Rayleigh block fading channel with CSI estimation
import tensorflow as tf
tf.config.run_functions_eagerly(True)
import numpy as np
from sionna.channel import OFDMChannel, RayleighBlockFading
from sionna.mimo import StreamManagement
from sionna.ofdm import ResourceGrid, ResourceGridMapper, KroneckerPilotPattern, LSChannelEstimator
from sionna.mapping import Mapper, Demapper
from sionna.utils import ebnodb2no, BinarySource, compute_ber
from jammer.pilots import PilotPatternWithSilence
from jammer.jammer import OFDMJammer
from jammer.mitigation.IAN import IanLMMSEEqualizer
from jammer.utils import covariance_estimation_from_signals

batch_size = 1 #TODO change
num_ofdm_symbols = 14
fft_size = 1024
num_ut = 4
num_ut_ant = 1
num_bs = 1
num_bs_ant = 18
num_jammer = 2
num_jammer_ant = 2
jammer_power = 1.
silent_symbols = np.arange(4)

no = ebnodb2no(10., 2, 1)
rg = ResourceGrid(num_ofdm_symbols, fft_size, 30e3, num_ut, num_ut_ant, pilot_pattern="kronecker", pilot_ofdm_symbol_indices=(4,))
rx_tx_association = np.zeros([1, num_ut])
rx_tx_association[0, :] = 1
stream_management = StreamManagement(rx_tx_association, num_ut_ant)
ut_bs_channel = RayleighBlockFading(num_bs, num_bs_ant, num_ut, num_ut_ant)
ofdm_channel = OFDMChannel(ut_bs_channel, rg)

rg.pilot_pattern = PilotPatternWithSilence(rg.pilot_pattern, silent_symbols)
jammer_bs_channel = RayleighBlockFading(num_bs, num_bs_ant, num_jammer, num_jammer_ant)
jammer = OFDMJammer(jammer_bs_channel, rg, num_jammer, num_jammer_ant)
equalizer = IanLMMSEEqualizer(rg, stream_management)

b = BinarySource()([batch_size, num_ut, num_ut_ant, rg.num_data_symbols * 2])
x = Mapper("qam", 2)(b)
x_rg = ResourceGridMapper(rg)(x)
y = ofdm_channel((x_rg, no))
y_jammed = jammer((y, jammer_power))
jammer_signals = tf.gather(y, silent_symbols, axis=3)
jammer_covariance = covariance_estimation_from_signals(jammer_signals, num_ofdm_symbols)
equalizer.set_jammer_covariance(jammer_covariance)
h_hat, err_var = LSChannelEstimator(rg)([y, no])
x_hat, no_eff = equalizer([y, h_hat, err_var, no])
b_hat = Demapper('app', 'qam', 2, hard_out=True)([x_hat, no_eff])
# ber = compute_ber(b, b_hat)
# print(ber)


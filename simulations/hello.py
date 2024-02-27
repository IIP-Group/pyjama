#%%
import os
# import drjit
gpu_num = 3 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sionna
import tensorflow as tf
import numpy as np
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
tf.get_logger().setLevel('ERROR')

from sionna.channel import OFDMChannel, RayleighBlockFading
from sionna.mimo import StreamManagement
from sionna.ofdm import ResourceGrid, ResourceGridMapper, KroneckerPilotPattern, LSChannelEstimator, LMMSEEqualizer
from sionna.mapping import Mapper, Demapper
from sionna.utils import ebnodb2no, BinarySource, compute_ber
from jammer.pilots import PilotPatternWithSilence
from jammer.jammer import OFDMJammer
from jammer.mitigation.IAN import IanLMMSEEqualizer
from jammer.mitigation.POS import OrthogonalSubspaceProjector
from jammer.utils import covariance_estimation_from_signals

# Jammer simulation over Rayleigh block fading channel with CSI estimation

# Normal OFDM init
batch_size = 64
num_ofdm_symbols = 14
fft_size = 1024
num_ut = 4
num_ut_ant = 1
num_bs = 1
num_bs_ant = 18
num_bits_per_symbol = 2

no = ebnodb2no(10., num_bits_per_symbol, 1)
rg = ResourceGrid(num_ofdm_symbols, fft_size, 30e3, num_ut, num_ut_ant,
                  pilot_pattern='kronecker', pilot_ofdm_symbol_indices=(4,))
rx_tx_association = np.zeros([1, num_ut])
rx_tx_association[0, :] = 1
stream_management = StreamManagement(rx_tx_association, num_ut_ant)
ut_bs_channel = RayleighBlockFading(num_bs, num_bs_ant, num_ut, num_ut_ant)
ofdm_channel = OFDMChannel(ut_bs_channel, rg)


# Jammer init
num_jammer = 2
num_jammer_ant = 2
jammer_power = 1.
silent_symbols = np.arange(4)

rg.pilot_pattern = PilotPatternWithSilence(rg.pilot_pattern, silent_symbols)
jammer_bs_channel = RayleighBlockFading(num_bs, num_bs_ant, num_jammer, num_jammer_ant)
jammer = OFDMJammer(jammer_bs_channel, rg, num_jammer, num_jammer_ant)
pos = OrthogonalSubspaceProjector()

# Simulation
b = BinarySource()([batch_size, num_ut, num_ut_ant, rg.num_data_symbols * num_bits_per_symbol])
x = Mapper('qam', num_bits_per_symbol)(b)
x_rg = ResourceGridMapper(rg)(x)
y = ofdm_channel((x_rg, no))

y_jammed = jammer((y, jammer_power))

jammer_signals = tf.gather(y_jammed, silent_symbols, axis=3)
pos.set_jammer_signals(jammer_signals)

y_mitigated = pos(y_jammed)

h_hat, err_var = LSChannelEstimator(rg)([y_mitigated, no])
x_hat, no_eff = LMMSEEqualizer(rg, stream_management)([y_mitigated, h_hat, err_var, no])
b_hat = Demapper('app', 'qam', 2, hard_out=True)([x_hat, no_eff])

ber = compute_ber(b, b_hat)
print(ber)


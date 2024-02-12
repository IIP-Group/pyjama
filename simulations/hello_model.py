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
# tf.config.run_functions_eagerly(True)

from jammer.simulation_model import *
from jammer.utils import *

# let's say we want to train a jammer in a coded UMi channel with 4UEs (1 Antenna each).
# The jammer is mitigated by a POS algorithm.
# The jammer has 2 antennas, and a power of 10dB/Antenna compared to a UE
# The jammer is estimated during the first 4 OFDM symbols.

model = Model(
    scenario = "umi",
    num_ut = 4,
    coderate = 0.5,
    jammer_present = True,
    perfect_csi=False,
    perfect_jammer_csi=False,
    jammer_mitigation = "pos",
    jammer_power = db_to_linear(10),
    jammer_parameters = {
        "num_tx_ant": 2,
        "trainable": True,
    },
    num_silent_pilot_symbols = 4,
    return_decoder_iterations = True,
)

train_model(
    model,
    num_iterations = 2000,
    loss_fn = negative_function(IterationLoss()),
    log_tensorboard = True,
    weights_filename = "weights.pickle"
)
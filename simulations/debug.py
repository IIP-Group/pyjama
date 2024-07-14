
import sys
gpu_num = int(sys.argv[1]) if len(sys.argv) > 1 else 1
parameter_num = int(sys.argv[2]) if len(sys.argv) > 2 else 0

import os
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
from tensorflow.python.keras.losses import MeanAbsoluteError, MeanSquaredError, BinaryCrossentropy

from pyjama.simulation_model import *
from pyjama.utils import *
import pyjama.simulation_model as sim

model_parameters = {}
jammer_parameters = {}
model_parameters["num_ut"] = 1
model_parameters["perfect_csi"] = True
model_parameters["num_silent_pilot_symbols"] = 8
jammer_parameters["num_tx"] = 2
jammer_parameters["num_tx_ant"] = 1
jammer_parameters["normalize_channel"] = True
model_parameters["jammer_parameters"] = jammer_parameters
model_parameters["scenario"] = "rayleigh"

model_parameters["perfect_jammer_csi"] = False
model_parameters["jammer_present"] = True
model_parameters["jammer_power"] = db_to_linear(10.)
model_parameters["jammer_mitigation"] = "pos"
model_parameters["jammer_mitigation_dimensionality"] = None

BATCH_SIZE = 2
model = Model(**model_parameters)
b, llr = model(BATCH_SIZE, 20.)
b_hat = sionna.utils.hard_decisions(llr)
ber = compute_ber(b, b_hat)
print(ber)
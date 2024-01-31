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
# tf.config.run_functions_eagerly(True)
from tensorflow.python.keras.losses import MeanAbsoluteError, MeanSquaredError, BinaryCrossentropy

from jammer.simulation_model import *
from jammer.utils import *
import jammer.simulation_model as sim
from simulations.experimental_losses import *

# common parameters
model_parameters = {}
jammer_parameters = {}
model_parameters["perfect_csi"] = False
# model_parameters["num_ut"] = 1
model_parameters["jammer_present"] = True
# model_parameters["jammer_power"] = 1.0
# model_parameters["jammer_mitigation"] = "pos"
# model_parameters["jammer_mitigation_dimensionality"] = 1
model_parameters["num_silent_pilot_symbols"] = 4
jammer_parameters["trainable"] = True
model_parameters["jammer_parameters"] = jammer_parameters
# changing but constant
# jammer_parameters["trainable_mask"] = tf.ones([14, 128], dtype=tf.bool)
jammer_parameters["trainable_mask"] = tf.ones([14, 1], dtype=tf.bool)

sim.BATCH_SIZE = 4

# massive grid: training with different jammer power and different number of UEs
num_ut = range(1, 9)
jammer_power = [db_to_linear(x) for x in np.arange(-2.5, 15.1, 2.5, dtype=np.float32)]
parameters = [(x, y) for x in num_ut for y in jammer_power]
n, p = parameters[parameter_num]
model_parameters["num_ut"] = n
model_parameters["jammer_power"] = p
model_parameters["num_ofdm_symbols"] = 14
model_parameters["fft_size"] = 64
model_parameters["num_bs_ant"] = 24
jammer_parameters["trainable_mask"] = tf.ones([14, 1], dtype=tf.bool)
model = Model(**model_parameters)
train_model(model,
            loss_fn=negative_function(MeanAbsoluteError()),
            loss_over_logits=False,
            weights_filename=f"weights/grid/ue_{n}_power_{linear_to_db(p):.1f}dB.pickle",
            log_tensorboard=True,
            log_weight_images=True,
            show_final_weights=False,
            num_iterations=2000,
            ebno_db=5.0)
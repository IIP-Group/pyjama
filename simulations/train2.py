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
model_parameters["num_ut"] = 1
model_parameters["jammer_present"] = True
model_parameters["jammer_power"] = 1.0
model_parameters["jammer_mitigation"] = "pos"
model_parameters["jammer_mitigation_dimensionality"] = 1
model_parameters["num_silent_pilot_symbols"] = 4
jammer_parameters["trainable"] = True
model_parameters["jammer_parameters"] = jammer_parameters
# changing but constant
# jammer_parameters["trainable_mask"] = tf.ones([14, 128], dtype=tf.bool)
jammer_parameters["trainable_mask"] = tf.ones([14, 1], dtype=tf.bool)

sim.BATCH_SIZE = 16


exponentials = [False, True]
alphas = np.arange(0.0, 1.1, 0.1, dtype=np.float32)
num_iters = [1, 2, 4, 8, 16]
parameters = [(x, y, z) for x in exponentials for y in alphas for z in num_iters]
exponential, alpha, num_iter = parameters[parameter_num]

model_parameters["num_ut"] = 1
model_parameters["decoder_parameters"] = {
    "num_iter": num_iter,
    "cn_type": "minsum"
}
model_parameters["return_decoder_iterations"] = False
model_parameters["coderate"] = 0.5
model = Model(**model_parameters)
filename = f"weights/coded/symbol/iteration_loss/ue_1_alpha_{alpha}_exp_{exponential}_{num_iter}_iter.pickle"
tensorboard_name = filename.split("/")[-1].rsplit(".", 1)[0]
# print(filename)
# print(tensorboard_name)
load_weights(model, filename)
tensorboard_validate_model(model, tensorboard_name)
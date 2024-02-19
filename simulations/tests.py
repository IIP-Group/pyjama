#%%
import os
# import drjit
gpu_num = 0 # Use "" to use the CPU
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

from jammer.simulation_model import *
from jammer.utils import *
import jammer.simulation_model as sim
from tensorflow.python.keras.losses import MeanAbsoluteError, MeanSquaredError, BinaryCrossentropy

# common parameters
model_parameters = {}
jammer_parameters = {}
model_parameters["scenario"] = "umi"
model_parameters["perfect_csi"] = False
model_parameters["num_ut"] = 2
model_parameters["num_ut_ant"] = 2
model_parameters["jammer_present"] = True
model_parameters["jammer_power"] = db_to_linear(-3.)
# model_parameters["jammer_mitigation"] = "pos"
# model_parameters["jammer_mitigation_dimensionality"] = 1
model_parameters["num_silent_pilot_symbols"] = 4
model_parameters["jammer_parameters"] = jammer_parameters

sim.BATCH_SIZE = 8
sim.MAX_MC_ITER = 50

model_parameters["jammer_present"] = True
model_parameters["mash"] = False
model = Model(**model_parameters)
simulate_model(model, "jammer")

model_parameters["jammer_present"] = True
model_parameters["mash"] = True
model = Model(**model_parameters)
simulate_model(model, "jammer, MASH")

ber_plots()
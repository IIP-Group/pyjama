# %%
import os
import drjit
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

from jammer.simulation_model import *
import jammer.simulation_model as sim
from tensorflow.python.keras.losses import MeanAbsoluteError, MeanSquaredError, BinaryCrossentropy

strategy = tf.distribute.MirroredStrategy()

def computation():
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

    ber_plots.reset()
    sim.MAX_MC_ITER = 100
    sim.BATCH_SIZE = 16
    jammer_parameters["trainable"] = False
    jammer_parameters["trainable_mask"] = tf.ones([14, 128])
    snrs = np.arange(-5.0, 10.5, 2.5)
    for snr in snrs:
        filename = f"weights/{snr}dB_relufix.pickle"
        model = Model(**model_parameters)
        load_weights(model, filename)
        simulate_model(model, f"{snr}dB, abs.")
    for snr in snrs:
        filename = f"weights/{snr}dB_quadratic_0.5co.pickle"
        model = Model(**model_parameters)
        load_weights(model, filename)
        simulate_model(model, f"{snr}dB, quadratic cutoff")
    ber_plots.title = "Absolute vs. Quadratic Cutoff (0.5) Loss"
    ber_plots()

strategy.run(computation)
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

from pyjama.simulation_model import *
from pyjama.utils import *
import pyjama.simulation_model as sim
from simulations.experimental_losses import *

# # common parameters
# model_parameters = {}
# jammer_parameters = {}
# model_parameters["perfect_csi"] = False
# model_parameters["jammer_present"] = True
# model_parameters["num_silent_pilot_symbols"] = 0
# jammer_parameters["trainable"] = True
# model_parameters["jammer_parameters"] = jammer_parameters
# # changing but constant
# jammer_parameters["trainable_mask"] = tf.ones([14, 1], dtype=tf.bool)

# sim.BATCH_SIZE = 2

# # massive grid: training with different jammer power and different number of UEs
# num_ut = range(1, 9)
# # jammer_power = [db_to_linear(x) for x in np.arange(-2.5, 15.1, 2.5, dtype=np.float32)]
# jammer_power = [db_to_linear(x) for x in np.arange(-2.5, 5.1, 2.5, dtype=np.float32)]
# parameters = [(x, y) for x in num_ut for y in jammer_power]
# n, p = parameters[parameter_num]
# model_parameters["num_ut"] = n
# model_parameters["jammer_power"] = p
# model_parameters["num_ofdm_symbols"] = 14
# model_parameters["fft_size"] = 64
# model_parameters["num_bs_ant"] = 24
# jammer_parameters["trainable_mask"] = tf.ones([14, 1], dtype=tf.bool)
# model = Model(**model_parameters)
# train_model(model,
#             loss_fn=negative_function(MeanAbsoluteError()),
#             loss_over_logits=False,
#             weights_filename=f"weights/unmitigated/grid/ue_{n}_power_{linear_to_db(p):.1f}dB.pickle",
#             log_tensorboard=True,
#             log_weight_images=True,
#             show_final_weights=False,
#             num_iterations=5000,
#             ebno_db=0.0)

import pickle
# ber_plots.reset()
# model_parameters = {}
# jammer_parameters = {}
# model_parameters["jammer_parameters"] = jammer_parameters
# model_parameters["num_ut"] = 1
# model_parameters["perfect_csi"] = False
# model_parameters["perfect_jammer_csi"] = False
# model_parameters["num_silent_pilot_symbols"] = 4
# jammer_parameters["num_tx_ant"] = 2
# model_parameters["jammer_present"] = True
# model_parameters["jammer_power"] = db_to_linear(5.)
# sim.BATCH_SIZE = 256
# sim.MAX_MC_ITER = 1000
# sim.ebno_dbs = np.linspace(-5., 15., 21)

# model = Model(**model_parameters)
# simulate_model(model, "Jammer, unmigitated")

# model_parameters["jammer_mitigation"] = "pos"
# model_parameters["jammer_mitigation_dimensionality"] = 2
# kmhs = [0, 20, 120]

# kmh = kmhs[parameter_num]
# meter_per_second = kmh / 3.6
# model_parameters["min_ut_velocity"] = meter_per_second
# model_parameters["max_ut_velocity"] = meter_per_second
# model_parameters["min_jammer_velocity"] = meter_per_second
# model_parameters["max_jammer_velocity"] = meter_per_second
# model = Model(**model_parameters)
# simulate_model(model, f"Jammer, POS, {kmh} km/h")

# ber_plots.title = "Jammers with velocity: Est. CSI, 1 UE, 1x2 Jammer (5dB/Ant)"
# # ber_plots(ylim=(1e-5, 1))
# with open(f"bers/velocities_{kmh}kmh.pickle", 'rb') as f:
#     bers = pickle.dump(ber_plots, f)

    

# ber_plots.reset()
# model_parameters = {}
# jammer_parameters = {}
# model_parameters["jammer_parameters"] = jammer_parameters
# model_parameters["num_ut"] = 1
# model_parameters["perfect_csi"] = False
# model_parameters["perfect_jammer_csi"] = False
# model_parameters["num_silent_pilot_symbols"] = 4
# jammer_parameters["num_tx_ant"] = 2
# sim.BATCH_SIZE = 256
# sim.MAX_MC_ITER = 1000
# sim.ebno_dbs = np.linspace(-5., 15., 21)

# # if parameter_num == 0:
# #     model = Model(**model_parameters)
# #     simulate_model(model, "No Jammer")

# # if parameter_num == 1:
# #     model_parameters["jammer_present"] = True
# #     model_parameters["jammer_power"] = db_to_linear(5.)
# #     model = Model(**model_parameters)
# #     simulate_model(model, "Jammer, unmigitated")

# if parameter_num == 2:
#     model_parameters["jammer_present"] = True
#     model_parameters["jammer_power"] = db_to_linear(15.)
#     model_parameters["jammer_mitigation"] = "pos"
#     model_parameters["jammer_mitigation_dimensionality"] = 2
#     model = Model(**model_parameters)
#     simulate_model(model, "Jammer, POS")

# ber_plots.title = "Simple Jammer Mitigation: Est. CSI, 1 UE, 1x2 Jammer (5dB/Ant)"
# ber_plots(ylim=(1e-5, 1))
# with open("bers/simple_pos.pickle", 'rb') as f:
#     bers = pickle.dump(ber_plots, f)



# because sombody killed it: learning gains ber
sim.EBN0_DB_MIN = -10
sim.EBN0_DB_MAX = 15
sim.NUM_SNR_POINTS = 26
# sim.BATCH_SIZE = 512
# sim.MAX_MC_ITER = 1500
sim.BATCH_SIZE = 512
sim.MAX_MC_ITER = 1000
sim.ebno_dbs = np.linspace(sim.EBN0_DB_MIN, sim.EBN0_DB_MAX, sim.NUM_SNR_POINTS)

# Learning gain BER
# symbol weights (old) vs. rg weights
model_parameters = {}
jammer_parameters = {}
model_parameters["perfect_csi"] = False
model_parameters["jammer_present"] = True
jammer_parameters["trainable"] = True
model_parameters["jammer_parameters"] = jammer_parameters

ber_plots.reset()
jammer_powers_db = [-5, 0, 10]
equivalent_jammer_powers = [-2.5, 1.5, 30.0]
for jammer_power_db, equivalent in zip(jammer_powers_db, equivalent_jammer_powers):
    print(f"Jammer power: {jammer_power_db}dB")
    print("Barrage")
    # barrage simulation
    model = Model(**model_parameters, num_ut=4, jammer_power=db_to_linear(equivalent))
    simulate_model(model, f"Barrage, {equivalent}dB", verbose=False)
    print("Learned symbol weights")
    # learned symbol weights
    jammer_parameters["trainable_mask"] = tf.ones([14, 1], dtype=tf.bool)
    model = Model(**model_parameters, num_ut=4, jammer_power=db_to_linear(jammer_power_db))
    load_weights(model, f"weights/unmitigated/symbol/ue_4_pow_{jammer_power_db}dB.pickle")
    simulate_model(model, f"Learned, symbol, {jammer_power_db}dB", verbose=False)
    print("Learned RG weights")
    # learned rg weights
    jammer_parameters["trainable_mask"] = tf.ones([14, 128], dtype=tf.bool)
    model = Model(**model_parameters, num_ut=4, jammer_power=db_to_linear(jammer_power_db))
    load_weights(model, f"weights/paper/unmitigated_rg_ue_4_pow_{jammer_power_db}dB.pickle")
    simulate_model(model, f"Learned, RG, {jammer_power_db}dB", verbose=False)

ber_plots()
with open("bers/paper/learning/learning_gains_ber.pickle", "wb") as f:
    pickle.dump(ber_plots, f)
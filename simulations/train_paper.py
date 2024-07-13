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
from tensorflow.python.keras.losses import MeanAbsoluteError, MeanSquaredError

from jammer.simulation_model import *
from jammer.utils import *
import jammer.simulation_model as sim

sim.EBN0_DB_MIN = -10
sim.EBN0_DB_MAX = 15
sim.NUM_SNR_POINTS = 26
# sim.BATCH_SIZE = 512
# sim.MAX_MC_ITER = 1500
sim.BATCH_SIZE = 512
sim.MAX_MC_ITER = 1000
sim.ebno_dbs = np.linspace(sim.EBN0_DB_MIN, sim.EBN0_DB_MAX, sim.NUM_SNR_POINTS)

# if parameter_num == 0:
#     sim.MAX_MC_ITER = 3000
#     ber_plots.reset()
#     model_parameters = {}
#     jammer_parameters = {}
#     decoder_parameters={}
#     model_parameters["jammer_parameters"] = jammer_parameters
#     model_parameters["decoder_parameters"] = decoder_parameters

#     model_parameters["num_silent_pilot_symbols"] = 4
#     jammer_parameters["num_tx"] = 2
#     jammer_parameters["num_tx_ant"] = 2
#     model_parameters["coderate"] = 0.5
#     decoder_parameters["cn_type"] = "minsum"
#     decoder_parameters["num_iter"] = 20

#     print("No jammer")
#     model = Model(**model_parameters)
#     model._decoder.llr_max = 1000
#     simulate_model(model, f"No jammer", add_bler=True, verbose=False)

#     print("Unmitigated jammer")
#     model_parameters["jammer_present"] = True
#     model = Model(**model_parameters)
#     model._decoder.llr_max = 1000
#     simulate_model(model, "Unmitigated jammer", add_bler=True, verbose=False)

#     print("Jammer, POS, 1D Nulling")
#     d = 4
#     model_parameters["jammer_mitigation"] = "pos"
#     model_parameters["jammer_mitigation_dimensionality"] = d
#     model = Model(**model_parameters)
#     simulate_model(model, f"Jammer, POS, {d}D Nulling", add_bler=True, verbose=False)

#     ber_plots.title = "Different Decoder Parameters: 4UE, 0.5 coderate, 0dB Jammer"
#     # ber_plots(show_ber=False)
#     with open("bers/paper/frequency/mitigation_bler.pickle", "wb") as f:
#         pickle.dump(ber_plots, f)
#     sim.MAX_MC_ITER = 1500

# if parameter_num == 1:
#     ber_plots.reset()
#     model_parameters = {}
#     jammer_parameters = {}
#     model_parameters["jammer_parameters"] = jammer_parameters

#     model_parameters["num_silent_pilot_symbols"] = 4
#     model_parameters["jammer_present"] = True
#     model_parameters["jammer_power"] = db_to_linear(20.)

#     print("Jammer, unmigitated, 0km/h")
#     model = Model(**model_parameters)
#     simulate_model(model, "Jammer, unmigitated, 0km/h", verbose=False)

#     model_parameters["jammer_mitigation"] = "pos"
#     model_parameters["jammer_mitigation_dimensionality"] = 1
#     kmhs = [0, 30, 80, 120]
#     for kmh in kmhs:
#         print("UE velocity: ", kmh)
#         meter_per_second = kmh / 3.6
#         model_parameters["min_ut_velocity"] = meter_per_second
#         model_parameters["max_ut_velocity"] = meter_per_second
#         # model_parameters["min_jammer_velocity"] = meter_per_second
#         # model_parameters["max_jammer_velocity"] = meter_per_second
#         model = Model(**model_parameters)
#         simulate_model(model, f"Jammer, POS, {kmh} km/h", verbose=False)

#     ber_plots.title = "UE velocity: Est. CSI, 1 UE, 1x1 Jammer (20dB)"
#     # ber_plots()
#     with open("bers/paper/frequency/ut_velocity_mitigation.pickle", "wb") as f:
#         pickle.dump(ber_plots, f)

# if parameter_num == 2:
#     ber_plots.reset()
#     model_parameters = {}
#     jammer_parameters = {}
#     model_parameters["jammer_parameters"] = jammer_parameters

#     model_parameters["num_silent_pilot_symbols"] = 4
#     model_parameters["jammer_present"] = True
#     model_parameters["jammer_power"] = db_to_linear(20.)

#     print("Jammer, unmigitated, 0km/h")
#     model = Model(**model_parameters)
#     simulate_model(model, "Jammer, unmigitated, 0km/h", verbose=False)

#     model_parameters["jammer_mitigation"] = "pos"
#     model_parameters["jammer_mitigation_dimensionality"] = 1
#     kmhs = [0, 30, 80, 120]
#     for kmh in kmhs:
#         print("Jammer velocity: ", kmh)
#         meter_per_second = kmh / 3.6
#         # model_parameters["min_ut_velocity"] = meter_per_second
#         # model_parameters["max_ut_velocity"] = meter_per_second
#         model_parameters["min_jammer_velocity"] = meter_per_second
#         model_parameters["max_jammer_velocity"] = meter_per_second
#         model = Model(**model_parameters)
#         simulate_model(model, f"Jammer, POS, {kmh} km/h", verbose=False)

#     ber_plots.title = "Jammer velocity: Est. CSI, 1 UE, 1x1 Jammer (20dB)"
#     # ber_plots()
#     with open("bers/paper/frequency/jammer_velocity_mitigation.pickle", "wb") as f:
#         pickle.dump(ber_plots, f)



# # Gian experiment
# # common parameters
# model_parameters = {}
# jammer_parameters = {}
# model_parameters["jammer_parameters"] = jammer_parameters

# model_parameters["jammer_present"] = True
# jammer_parameters["trainable"] = True
# jammer_parameters["trainable_mask"] = tf.ones([14, 1], dtype=tf.bool)
# model_parameters["num_ut"] = 2
# model_parameters["num_ofdm_symbols"] = 14
# model_parameters["jammer_power"] = db_to_linear(10.)

# sim.BATCH_SIZE = 2

# if parameter_num == 0:
#     model_parameters["num_silent_pilot_symbols"] = 0
#     filename = "weights/paper/unmitigated_ue_2_power_10.0dB.pickle"
# elif parameter_num == 1:
#     model_parameters["num_silent_pilot_symbols"] = 2
#     model_parameters["jammer_mitigation"] = "pos"
#     filename = "weights/paper/pos_ue_2_power_10.0dB.pickle"

# model = Model(**model_parameters)
# train_model(model,
#             loss_fn=negative_function(MeanAbsoluteError()),
#             loss_over_logits=False,
#             weights_filename=filename,
#             log_tensorboard=True,
#             log_weight_images=True,
#             show_final_weights=False,
#             num_iterations=5000,
#             ebno_db=5.0)


# Comparison trained/uniform: training (unmitigated)
# model_parameters = {}
# jammer_parameters = {}
# model_parameters["jammer_parameters"] = jammer_parameters

# model_parameters["jammer_present"] = True
# jammer_parameters["trainable"] = True
# jammer_parameters["trainable_mask"] = tf.ones([14, 128], dtype=tf.bool)
# model_parameters["num_ut"] = 4
# num_ut = 4
# model_parameters["num_ut"] = num_ut

# # jammer_powers_db = [-5, 10]
# parameters = [(-5, -5.), (0,0.), (5,0.), (10, 2.5)]
# jammer_power_db, ebno_db = parameters[parameter_num]
# model_parameters["jammer_power"] = db_to_linear(jammer_power_db)

# model = Model(**model_parameters)
# train_model(model,
#             loss_fn=negative_function(MeanAbsoluteError()),
#             loss_over_logits=False,
#             weights_filename=f"weights/paper/unmitigated_rg_ue_{num_ut}_pow_{jammer_power_db}dB.pickle",
#             log_tensorboard=True,
#             log_weight_images=True,
#             show_final_weights=False,
#             num_iterations=30000,
#             ebno_db=ebno_db,
#             validate_ber_tensorboard=True)



# Learning gain BER
# symbol weights (old) vs. rg weights
# model_parameters = {}
# jammer_parameters = {}
# model_parameters["perfect_csi"] = False
# model_parameters["jammer_present"] = True
# jammer_parameters["trainable"] = True
# model_parameters["jammer_parameters"] = jammer_parameters

# ber_plots.reset()
# jammer_powers_db = [-5, 0, 10]
# equivalent_jammer_powers = [-2.5, 1.5, 30.0]
# for jammer_power_db, equivalent in zip(jammer_powers_db, equivalent_jammer_powers):
#     print(f"Jammer power: {jammer_power_db}dB")
#     print("Barrage")
#     # barrage simulation
#     model = Model(**model_parameters, num_ut=4, jammer_power=db_to_linear(equivalent))
#     simulate_model(model, f"Barrage, {equivalent}dB", verbose=False)
#     print("Learned symbol weights")
#     # learned symbol weights
#     jammer_parameters["trainable_mask"] = tf.ones([14, 1], dtype=tf.bool)
#     model = Model(**model_parameters, num_ut=4, jammer_power=db_to_linear(jammer_power_db))
#     load_weights(model, f"weights/unmitigated/symbol/ue_4_pow_{jammer_power_db}dB.pickle")
#     simulate_model(model, f"Learned, symbol, {jammer_power_db}dB", verbose=False)
#     print("Learned RG weights")
#     # learned rg weights
#     jammer_parameters["trainable_mask"] = tf.ones([14, 128], dtype=tf.bool)
#     model = Model(**model_parameters, num_ut=4, jammer_power=db_to_linear(jammer_power_db))
#     load_weights(model, f"weights/paper/unmitigated_rg_ue_4_pow_{jammer_power_db}dB.pickle")
#     simulate_model(model, f"Learned, RG, {jammer_power_db}dB", verbose=False)

# ber_plots()
# with open("bers/paper/learning/learning_gains_ber.pickle", "wb") as f:
#     pickle.dump(ber_plots, f)



# coded: learned vs uniform
# sim.MAX_MC_ITER = 3000
# ber_plots.reset()
# # common parameters
# model_parameters = {}
# jammer_parameters = {}
# decoder_parameters={}
# model_parameters["num_ut"] = 4
# model_parameters["jammer_present"] = True
# model_parameters["coderate"] = 0.5
# jammer_parameters["trainable"] = False
# jammer_parameters["trainable_mask"] = tf.ones([14, 128], dtype=tf.bool)
# decoder_parameters["num_iter"] = 20
# model_parameters["jammer_parameters"] = jammer_parameters
# model_parameters["decoder_parameters"] = decoder_parameters

# # Uniform jammer with Minsum
# decoder_parameters["cn_type"] = "minsum"
# model = Model(**model_parameters)
# model._decoder.llr_max = 1000
# simulate_model(model, "Uniform Jammer, Minsum", add_bler=True)

# # Trained jammer
# model = Model(**model_parameters)
# model._decoder.llr_max = 1000
# load_weights(model, f"weights/paper/ue_4_coded.pickle")
# simulate_model(model, "Trained Jammer, Minsum", add_bler=True)

# ber_plots.title = "Trained Jammer vs. Uniform Jammer"
# # ber_plots(show_ber=False)
# # TODO save the plot
# with open("bers/paper/coded_bler.pickle", 'wb') as f:
#     bers = pickle.dump(ber_plots, f)



# # grid retraining
# common parameters
model_parameters = {}
jammer_parameters = {}
model_parameters["jammer_parameters"] = jammer_parameters
model_parameters["jammer_present"] = True
jammer_parameters["trainable"] = True
jammer_parameters["trainable_mask"] = tf.ones([14, 1], dtype=tf.bool)
sim.BATCH_SIZE = 2
# massive grid: training with different jammer power and different number of UEs
runs = [1, 2]
num_ut = range(1, 9)
jammer_powers_db = np.arange(-5, 15.1, 5, dtype=np.float32)
parameters = [(r, x, y) for r in runs for x in num_ut for y in jammer_powers_db]
r, n, p = parameters[parameter_num]
model_parameters["num_ut"] = n
model_parameters["jammer_power"] = db_to_linear(p)
model = Model(**model_parameters)
ebno_db = 5.0 if p > 2.5 else 0.
train_model(model,
            loss_fn=negative_function(MeanAbsoluteError()),
            loss_over_logits=False,
            weights_filename=f"weights/paper/grid_exp/{r}/ue_{n}_power_{p:.1f}dB.pickle",
            log_tensorboard=True,
            log_weight_images=True,
            show_final_weights=False,
            num_iterations=20000,
            ebno_db=ebno_db)
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
# jammer_parameters["trainable_mask"] = tf.ones([14, 1], dtype=tf.bool)

sim.BATCH_SIZE = 2
# sim.BATCH_SIZE = 1


# training on different loss functions
# # needed, as keras MSE does not take |.|
# abs_mse = lambda y_true, y_pred: tf.reduce_mean(tf.square(tf.abs(y_true - y_pred)))
# abs_log = lambda y_true, y_pred: tf.reduce_mean(tf.math.log(tf.abs(y_true - y_pred) + 1))
# # name, loss_fn, over symbols?, loss_over_logits
# parameters = [
#     ("L1 over symbols", negative_function(MeanAbsoluteError()), True, False),
#     ("MSE over symbols", negative_function(abs_mse), True, False),
#     ("L1 over bit estimates", negative_function(MeanAbsoluteError()), False, False),
#     ("MSE over bit estimates", negative_function(MeanSquaredError()), False, False),
#     ("BCE over bit estimates (logits)", BinaryCrossentropy(from_logits=True), False, True),
#     ("log over bit estimates", negative_function(abs_log), False, False),
# ]
# name, loss_fn, over_symbols, loss_over_logits = parameters[parameter_num]
# model = Model(**model_parameters, return_symbols=over_symbols)
# train_model(model,
#             loss_fn=loss_fn,
#             loss_over_logits=loss_over_logits,
#             weights_filename=f"weights/{name} symbol_weights.pickle",
#             log_tensorboard=True,
#             log_weight_images=True,
#             show_final_weights=False,
#             num_iterations=2000,
#             ebno_db=0.0)


# # different SNRs
# parameters = np.arange(-2.5, 10.5, 2.5, dtype=np.float32)
# jammer_parameters["training_constraint"] = MaxMeanSquareNorm(1.0)
# jammer_parameters["constraint_integrated"] = False
# model = Model(**model_parameters)
# train_model(model,
#             loss_fn=negative_function(MeanAbsoluteError()),
#             loss_over_logits=False,
#             weights_filename=f"weights/{parameters[parameter_num]}dB_relufix_constraint.pickle",
#             log_tensorboard=True,
#             log_weight_images=True,
#             show_final_weights=False,
#             num_iterations=2000,
#             ebno_db=parameters[parameter_num])

# # different number of UEs
# parameters = np.arange(1, 5, dtype=np.int32)
# model_parameters["num_ut"] = parameters[parameter_num]
# model = Model(**model_parameters)
# train_model(model,
#             loss_fn=negative_function(MeanAbsoluteError()),
#             loss_over_logits=False,
#             weights_filename=f"weights/ue_{parameters[parameter_num]}_relufix_symbol_weights.pickle",
#             log_tensorboard=True,
#             log_weight_images=True,
#             show_final_weights=False,
#             num_iterations=2000,
#             ebno_db=0.0)


# # different SNRs, symbol learning 4ue
# model_parameters["num_ut"] = 4
# parameters = np.arange(-2.5, 10.5, 2.5, dtype=np.float32)
# model = Model(**model_parameters)
# train_model(model,
#             loss_fn=negative_function(MeanAbsoluteError()),
#             loss_over_logits=False,
#             weights_filename=f"weights/ue_4_{parameters[parameter_num]}dB.pickle",
#             log_tensorboard=True,
#             log_weight_images=True,
#             show_final_weights=False,
#             num_iterations=2000,
#             ebno_db=parameters[parameter_num])


# massive grid: training with different jammer power and different number of UEs
# num_ut = range(1, 9)
# jammer_power = [db_to_linear(x) for x in np.arange(-2.5, 15.1, 2.5, dtype=np.float32)]
# parameters = [(x, y) for x in num_ut for y in jammer_power]
# n, p = parameters[parameter_num]
# model_parameters["num_ut"] = n
# model_parameters["jammer_power"] = p
# model_parameters["num_ofdm_symbols"] = 18
# model_parameters["fft_size"] = 64
# jammer_parameters["trainable_mask"] = tf.ones([18, 1], dtype=tf.bool)
# model = Model(**model_parameters)
# train_model(model,
#             loss_fn=negative_function(MeanAbsoluteError()),
#             loss_over_logits=False,
#             weights_filename=f"weights/grid/ue_{n}_power_{linear_to_db(p):.1f}dB.pickle",
#             log_tensorboard=True,
#             log_weight_images=True,
#             show_final_weights=False,
#             num_iterations=2000,
#             ebno_db=5.0)

# # different SNRs, experimental loss
# parameters = np.arange(-5.0, 10.5, 2.5, dtype=np.float32)
# model = Model(**model_parameters)
# train_model(model,
#             loss_fn=negative_function(QuadraticCutoffLoss()),
#             loss_over_logits=False,
#             weights_filename=f"weights/{parameters[parameter_num]}dB_quadratic.pickle",
#             log_tensorboard=True,
#             log_weight_images=True,
#             show_final_weights=False,
#             num_iterations=2000,
#             ebno_db=parameters[parameter_num])

# # different SNRs, experimental loss
# model = Model(**model_parameters)
# train_model(model,
#             loss_fn=negative_function(QuadraticCutoffLoss()),
#             loss_over_logits=False,
#             weights_filename=f"weights/{parameters[parameter_num]}dB_quadratic.pickle",
#             log_tensorboard=True,
#             log_weight_images=True,
#             show_final_weights=False,
#             num_iterations=2000,
#             ebno_db=0.0)

# # different number of UEs, experimental loss
# parameters = np.arange(1, 5, dtype=np.int32)
# model_parameters["num_ut"] = parameters[parameter_num]
# model = Model(**model_parameters)
# train_model(model,
#             loss_fn=negative_function(MeanAbsoluteError()),
#             loss_over_logits=False,
#             weights_filename=f"weights/ue_{parameters[parameter_num]}_quadratic_symbol_weights.pickle",
#             log_tensorboard=True,
#             log_weight_images=True,
#             show_final_weights=False,
#             num_iterations=2000,
#             ebno_db=0.0)

# 1 & 4 UEs, trained on coded channel information bits
# num_ut, num_iter (decoder), cn_type (decoder)
# num_uts = [1, 4]
# num_iters = [8, 12, 16]
# cn_types = ["boxplus-phi", "minsum"]
# parameters = [(x, y, z) for x in num_uts for y in num_iters for z in cn_types]

# num_ut = parameters[parameter_num][0]
# num_iter = parameters[parameter_num][1]
# cn_type = parameters[parameter_num][2]
# model_parameters["num_ut"] = num_ut
# model_parameters["decoder_parameters"] = {
#     "num_iter": num_iter,
#     "cn_type": cn_type
# }
# model_parameters["coderate"] = 0.5
# model = Model(**model_parameters)
# train_model(model,
#             loss_fn=negative_function(MeanAbsoluteError()),
#             loss_over_logits=False,
#             weights_filename=f"weights/coded/symbol/ue_{num_ut}_{cn_type}_{num_iter}_iter.pickle",
#             log_tensorboard=True,
#             log_weight_images=True,
#             show_final_weights=False,
#             num_iterations=3000,
#             ebno_db=0.0,
#             validate_ber_tensorboard=True)



# exponentials = [False, True]
# alphas = np.arange(0.0, 1.1, 0.1, dtype=np.float32)
# num_iters = [1, 2, 4, 8, 16]
# parameters = [(x, y, z) for x in exponentials for y in alphas for z in num_iters]
# exponential, alpha, num_iter = parameters[parameter_num]

# model_parameters["num_ut"] = 1
# model_parameters["decoder_parameters"] = {
#     "num_iter": num_iter,
#     "cn_type": "minsum"
# }
# model_parameters["return_decoder_iterations"] = True
# model_parameters["coderate"] = 0.5
# model = Model(**model_parameters)
# loss = IterationLoss(alpha=alpha, exponential_alpha_scaling=exponential)
# train_model(model,
#             loss_fn=negative_function(loss),
#             loss_over_logits=False,
#             weights_filename=f"weights/coded/symbol/iteration_loss/ue_1_alpha_{alpha}_exp_{exponential}_{num_iter}_iter.pickle",
#             log_tensorboard=True,
#             log_weight_images=True,
#             show_final_weights=False,
#             num_iterations=2000,
#             ebno_db=0.0)


# Iteration loss, 1&4 UEs
# num_uts = [1, 4]
# exponentials = [False, True]
# num_iters = [8, 12, 16, 20]
# alphas = np.arange(0.1, 1.0, 0.2, dtype=np.float32)
# parameters = [(w, x, y, z) for w in num_uts for x in exponentials for y in num_iters for z in alphas]
# num_ut, exponential, num_iter, alpha = parameters[parameter_num]

# model_parameters["num_ut"] = num_ut
# model_parameters["decoder_parameters"] = {
#     "num_iter": num_iter,
#     "cn_type": "minsum"
# }
# model_parameters["return_decoder_iterations"] = True
# model_parameters["coderate"] = 0.5
# model = Model(**model_parameters)
# loss = IterationLoss(alpha=alpha, exponential_alpha_scaling=exponential)
# train_model(model,
#             loss_fn=negative_function(loss),
#             loss_over_logits=False,
#             weights_filename=f"weights/coded/symbol/iteration_loss_2/ue_{num_ut}_alpha_{alpha:.1}_exp_{exponential}_{num_iter}_iter.pickle",
#             log_tensorboard=True,
#             log_weight_images=True,
#             show_final_weights=False,
#             num_iterations=5000,
#             ebno_db=0.0,
#             validate_ber_tensorboard=True)

# Iteration Loss Validation
# sim.BATCH_SIZE = 16
# exponentials = [False, True]
# alphas = np.arange(0.0, 1.1, 0.1, dtype=np.float32)
# num_iters = [1, 2, 4, 8, 16]
# parameters = [(x, y, z) for x in exponentials for y in alphas for z in num_iters]
# exponential, alpha, num_iter = parameters[parameter_num]
# model_parameters["num_ut"] = 1
# model_parameters["decoder_parameters"] = {
#     "num_iter": num_iter,
#     "cn_type": "minsum"
# }
# model_parameters["return_decoder_iterations"] = False
# model_parameters["coderate"] = 0.5
# model = Model(**model_parameters)
# filename = f"weights/coded/symbol/iteration_loss/ue_1_alpha_{alpha}_exp_{exponential}_{num_iter}_iter.pickle"
# tensorboard_name = filename.split("/")[-1].rsplit(".", 1)[0]
# load_weights(model, filename)
# tensorboard_validate_model(model, tensorboard_name)


# Unmitigated
# model_parameters["jammer_mitigation"] = None
# model_parameters["num_silent_pilot_symbols"] = 0
# num_uts = [1, 4]
# jammer_powers_db = [-5, 0, 5, 10]
# parameters = [(x, y) for x in num_uts for y in jammer_powers_db]
# num_ut, jammer_power_db = parameters[parameter_num]

# model_parameters["num_ut"] = num_ut
# model_parameters["jammer_power"] = db_to_linear(jammer_power_db)
# model = Model(**model_parameters)
# train_model(model,
#             loss_fn=negative_function(MeanAbsoluteError()),
#             loss_over_logits=False,
#             weights_filename=f"weights/unmitigated/symbol/ue_{num_ut}_pow_{jammer_power_db}dB.pickle",
#             log_tensorboard=True,
#             log_weight_images=True,
#             show_final_weights=False,
#             num_iterations=3000,
#             ebno_db=0.0,
#             validate_ber_tensorboard=True)

# # Unmitigated, tensorflow barrage validation to find dB difference of trained and barrage
# sim.BATCH_SIZE = 8
# model_parameters["jammer_mitigation"] = None
# model_parameters["num_silent_pilot_symbols"] = 0
# num_uts = [1, 4]
# # jammer_powers_db = [-5, 0, 5, 10]
# jammer_powers_db = np.arange(-5, 20, 0.25)
# parameters = [(x, y) for x in num_uts for y in jammer_powers_db]
# num_ut, jammer_power_db = parameters[parameter_num]

# model_parameters["num_ut"] = num_ut
# model_parameters["jammer_power"] = db_to_linear(jammer_power_db)
# model = Model(**model_parameters)
# import datetime
# current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
# name = f"barrage_ue_{num_ut}_pow_{jammer_power_db}dB"
# tensorboard_validate_model(model, 'logs/tensorboard/' + current_time + '-' + name)


# Train Uncoded and Coded Resource Grid for Visualization Purposes
# 1 & 4 UEs, trained on coded channel information bits
# jammer_parameters["trainable_mask"] = tf.ones([14, 128], dtype=tf.bool)
# # jammer_parameters["trainable_mask"] = tf.ones([14, 1], dtype=tf.bool)
# num_ut = 1
# num_iter = 4
# cn_type = "minsum"
# model_parameters["num_ut"] = num_ut
# model_parameters["decoder_parameters"] = {
#     "num_iter": num_iter,
#     "cn_type": cn_type,
# }
# model_parameters["jammer_mitigation"] = None
# model_parameters["num_silent_pilot_symbols"] = 0
# if parameter_num == 0:
#     model_parameters["coderate"] = None
#     loss = MeanAbsoluteError()
#     filename = f"weights/presentation/ue_{num_ut}_uncoded.pickle"
# else:
#     model_parameters["return_decoder_iterations"] = True
#     model_parameters["coderate"] = 0.5
#     loss = IterationLoss(alpha=0.5, exponential_alpha_scaling=False)
#     filename = f"weights/presentation/ue_{num_ut}_coded.pickle"

# model = Model(**model_parameters)
# if parameter_num == 1:
#     model._decoder.llr_max = 1000
# train_model(model,
#             learning_rate=0.001,
#             loss_fn=negative_function(loss),
#             loss_over_logits=False,
#             weights_filename=filename,
#             log_tensorboard=True,
#             log_weight_images=True,
#             show_final_weights=False,
#             num_iterations=20000,
#             ebno_db=0.0,
#             validate_ber_tensorboard=True)


# NonNegMaxMeanSquareNorm vs MaxMeanSquareNorm
# num_ues = np.arange(1, 5, dtype=np.int32)
# nonnegs = [True, False]
# parameters = [(x, y) for x in num_ues for y in nonnegs]
# num_ue = parameters[parameter_num][0]
# nonneg = parameters[parameter_num][1]

# model_parameters["num_ut"] = num_ue
# jammer_parameters["training_constraint"] = NonNegMaxMeanSquareNorm(1.0) if nonneg else MaxMeanSquareNorm(1.0)

# filename = f"weights/nonneg_vs_neg/ue_{num_ue}_nonneg_{nonneg}.pickle"
# model = Model(**model_parameters)
# train_model(model,
#             learning_rate=0.005,
#             weights_filename=filename,
#             log_tensorboard=True,
#             log_weight_images=True,
#             show_final_weights=False,
#             num_iterations=5000,
#             ebno_db=0.0,
#             validate_ber_tensorboard=True)

# symbol weights for comparison
num_ues = np.arange(1, 5, dtype=np.int32)
num_ue = num_ues[parameter_num]

model_parameters["num_ut"] = num_ue
jammer_parameters["trainable_mask"] = tf.ones([14, 1], dtype=tf.bool)

filename = f"weights/nonneg_vs_neg/ue_{num_ue}_symbol.pickle"
model = Model(**model_parameters)
train_model(model,
            learning_rate=0.001,
            weights_filename=filename,
            log_tensorboard=True,
            log_weight_images=True,
            show_final_weights=False,
            num_iterations=5000,
            ebno_db=0.0,
            validate_ber_tensorboard=True)
import sys
gpu_num = int(sys.argv[1]) if len(sys.argv) > 1 else 0
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
jammer_parameters["trainable_mask"] = tf.ones([14, 1], dtype=tf.bool)

sim.BATCH_SIZE = 1


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
parameters = np.arange(1, 5, dtype=np.int32)
model_parameters["num_ut"] = parameters[parameter_num]
model = Model(**model_parameters)
train_model(model,
            loss_fn=negative_function(MeanAbsoluteError()),
            loss_over_logits=False,
            weights_filename=f"weights/ue_{parameters[parameter_num]}_quadratic_symbol_weights.pickle",
            log_tensorboard=True,
            log_weight_images=True,
            show_final_weights=False,
            num_iterations=2000,
            ebno_db=0.0)
# %%
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
# tf.config.run_functions_eagerly(True)

from jammer.simulation_model import *
from tensorflow.python.keras.losses import MeanAbsoluteError, MeanSquaredError, BinaryCrossentropy


# common parameters
model_parameters["perfect_csi"] = False
model_parameters["jammer_present"] = True
model_parameters["jammer_mitigation"] = "pos"
model_parameters["jammer_mitigation_dimensionality"] = 1
jammer_parameters["trainable"] = True
model_parameters["return_symbols"] = True

abs_mse = lambda y_true, y_pred: tf.reduce_mean(tf.square(tf.abs(y_true - y_pred)))
# name, loss_fn, over symbols?, loss_over_logits
parameters = [
    ("L1 over symbols", negative_function(MeanAbsoluteError()), True, False),
    ("MSE over symbols", negative_function(abs_mse), True, False),
    ("MSE over bit estimates", negative_function(MeanSquaredError()), False, False),
    ("L1 over bit estimates", negative_function(MeanAbsoluteError()), False, False),
    ("BCE over bit estimates (logits)", BinaryCrossentropy(from_logits=True), False, True),
]

for name, loss_fn, over_symbols, loss_over_logits in parameters:
    model_parameters["return_symbols"] = over_symbols
    model = Model(**model_parameters)
    train_model(model,
                loss_fn=loss_fn,
                loss_over_logits=loss_over_logits,
                weights_filename=f"weights/{name}.pickle",
                log_tensorboard=True,
                log_weight_images=True,
                show_final_weights=True,
                num_iterations=20)
# %%

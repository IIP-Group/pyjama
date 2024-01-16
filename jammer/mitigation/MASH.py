"""Mitigation via Subspace Hiding (MASH)"""
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


import tensorflow as tf
import numpy as np
import sionna
from jammer.custom_pilots import PilotPatternWithSilence
# from ..utils import reduce_matrix_rank

class HaarApproximation:
    """Approximates a Haar matrix `A` as F@diag(E_1)@F@diag(E_2)... where F FFT and E_i vectors with values -1 or 1."""
    def __init__(self, n, num_iterations=3):
        self._n = n
        self._num_iterations = num_iterations
        # each row is a vector of length n with values -1 or 1
        self._diag_values = tf.where(tf.random.uniform((num_iterations, n)) > 0.5, tf.ones((num_iterations, n)), -tf.ones((num_iterations, n)))

    def multiply_from_right(self, x, adjoint=False):
        """Returns x @ A or x @ A^H."""
        if adjoint:
            # x @ A^H = (A @ x^H)^H
            x = tf.linalg.adjoint(x)
            for i in range(self._num_iterations):
                x = tf.linalg.diag(self._diag_values[i]) @ x
                x = tf.signal.fft(x)
        else:
            # x @ A = (A^H @ x^H)^H
            x = tf.linalg.adjoint(x)
            for i in range(self._num_iterations):
                x = tf.signal.ifft(x)
                x = tf.linalg.diag(self._diag_values[i]) @ x
        return tf.linalg.adjoint(x)

class MASH(tf.keras.layers.Layer):
    # C_orthogonal should take place at the jammer-training symbols (pilot-mask, those rows(?))
    # MASH has to be done after RG-Mapper, as pilots have to be mixed in. But, this means we have to filter out the guard and DC bands.
    def __init__(self, resource_grid):
        self._rg = resource_grid
        self._sc_ind = resource_grid.effective_subcarrier_ind
        # indices where for every tx and stream, the pilots are zero. List of indices.
        # as different streams might mask different resource elements, we have to map all patterns to the same resource grid.
        pilots = sionna.utils.flatten_last_dims(resource_grid.pilot_pattern.pilots, 3)
        pilot_indices = tf.where(resource_grid.pilot_pattern.mask)
        # -1 where no pilot is present, pilot value otherwise
        pilot_pattern_grid = tf.tensor_scatter_nd_update(tf.ones(resource_grid.pilot_pattern.mask.shape, dtype=pilots.dtype) * -1, pilot_indices, pilots)
        pilot_pattern_grid_vectorized = sionna.utils.flatten_last_dims(pilot_pattern_grid)
        # list of indices where all pilots are zero (1-dim) in vectorized resource grid.
        self._silent_pilot_indices = tf.where(tf.math.reduce_all(pilot_pattern_grid_vectorized == 0j, axis=(0, 1)))
    
    def call(self, inputs):
        """Input:
        [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex: The whole RG."""
        # remove guard and DC bands -> [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers]
        x = tf.gather(inputs, self._sc_ind, axis=-1)
        # find the jammer-training symbols


# num_ofdm_symbols = 8
# fft_size = 20
# subcarrier_spacing = 15e3
# num_tx = 2
# num_streams_per_tx = 2
# cyclic_prefix_length = 10

# rg = sionna.ofdm.ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
#                   fft_size=fft_size,
#                   subcarrier_spacing=subcarrier_spacing,
#                   num_tx=num_tx,
#                   num_streams_per_tx=num_streams_per_tx,
#                   cyclic_prefix_length=cyclic_prefix_length,
#                   pilot_pattern="kronecker",
#                   pilot_ofdm_symbol_indices = [2])
# rg.pilot_pattern = PilotPatternWithSilence(rg.pilot_pattern, [0, 3])
# # rg.pilot_pattern.show()

# mash = MASH(rg)
# print(mash._silent_pilot_indices)


# fft_matrix = tf.sqrt(tf.constant(0.2, dtype=tf.complex64)) * tf.signal.fft(tf.eye(5, dtype=tf.complex64))
# i = tf.matmul(fft_matrix, fft_matrix, adjoint_b=True)
# print(tf.complex(tf.round(tf.math.real(i)), tf.round(tf.math.imag(i))))
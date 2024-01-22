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
tf.config.run_functions_eagerly(True)

import tensorflow as tf
import numpy as np
import sionna
from sionna.utils import flatten_last_dims
from jammer.custom_pilots import PilotPatternWithSilence
# from ..utils import reduce_matrix_rank

class HaarApproximation:
    """Approximates a Haar matrix `A` as F@diag(E_1)@F@diag(E_2)... where F FFT and E_i vectors with values -1 or 1."""
    def __init__(self, n, num_iterations=3, dtype=tf.complex64):
        self._n = n
        self._norm_factor = tf.sqrt(tf.constant(1. / n, dtype=dtype))
        self._num_iterations = num_iterations
        # each row is a vector of length n with values -1 or 1
        self._diag_values = tf.where(tf.random.uniform((num_iterations, n)) > 0.5,
                                     tf.ones((num_iterations, n), dtype=dtype),
                                     -tf.ones((num_iterations, n), dtype=dtype))

    def multiply_from_right(self, x, adjoint=False):
        """Returns x @ A or x @ A^H."""
        if adjoint:
            # x @ A^H = (A @ x^H)^H
            x = tf.linalg.adjoint(x)
            for i in range(self._num_iterations-1, -1, -1):
                x = tf.linalg.diag(self._diag_values[i]) @ x
                x = self._norm_factor * tf.signal.fft(x)
        else:
            # x @ A = (A^H @ x^H)^H
            x = tf.linalg.adjoint(x)
            for i in range(self._num_iterations):
                x = 1./self._norm_factor * tf.signal.ifft(x)
                x = tf.linalg.diag(self._diag_values[i]) @ x
        return tf.linalg.adjoint(x)

class Mash(tf.keras.layers.Layer):
    """Intended to be used after RG-Mapper."""
    def __init__(self, resource_grid, renew_secret=True, **kwargs):
        super().__init__(**kwargs)
        self._rg = resource_grid
        self._renew_secret = renew_secret
        self._sc_ind = resource_grid.effective_subcarrier_ind
        self.C = None
        # self._silent_pilot_indices = tf.where(tf.math.reduce_all(resource_grid.pilot_pattern.pilots == 0j, axis=(0, 1)))
        # self._haar_approximation = HaarApproximation(resource_grid.num_effective_subcarriers)

    def call(self, inputs):
        """Input:
        [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex: The whole RG."""
        # remove guard and DC bands -> [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers]
        x = tf.gather(inputs, self._sc_ind, axis=-1)
        # flatten RG
        before_flatten_shape = x.shape
        x = flatten_last_dims(x, 2)
        n = x.shape[-1]
        if self._renew_secret or self.C is None:
            self.C = HaarApproximation(n)
        # TODO do we have to flatten num_tx and num_streams_per_tx? (I think not)
        # x = sionna.utils.flatten_dims(x, 2, 1)
        # x contains zeros at silent pilot indices. I.e. we can just multiply with C (and C^H) to demash.
        x = self.C.multiply_from_right(x)
        # reshape to RG without guard and DC bands
        x = tf.reshape(x, before_flatten_shape)
        # add guard and DC bands again
        x = tf.transpose(x, (4, 0, 1, 2, 3))
        indices = tf.expand_dims(self._sc_ind, -1)
        x = tf.scatter_nd(
            indices, x, tf.concat([[tf.shape(inputs, tf.int64)[-1]], tf.shape(inputs, tf.int64)[:-1]], axis=0))
        return tf.transpose(x, (1, 2, 3, 4, 0))
        

class DeMash(tf.keras.layers.Layer):
    def __init__(self, mash, **kwargs):
        super().__init__(**kwargs)
        self._mash = mash

    def call(self, inputs):
        """Input:
        [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex: The whole RG."""
        # remove guard and DC bands -> [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers]
        y = tf.gather(inputs, self._sc_ind, axis=-1)
        # flatten RG
        y = flatten_last_dims(y, 2)
        # x contains zeros at silent pilot indices. I.e. we can just multiply with C (and C^H) to demash.
        y = self._mash.C.multiply_from_right(y, adjoint=True)
        # add guard and DC bands again
        y = tf.transpose(y, (4, 0, 1, 2, 3))
        indices = tf.expand_dims(self._mash._sc_ind, -1)
        y = tf.scatter_nd(
            indices, y, tf.concat([[tf.shape(inputs, tf.int64)[-1]], tf.shape(inputs, tf.int64)[:-1]], axis=0))
        return tf.transpose(y, (1, 2, 3, 4, 0))

        

def test():
    num_ofdm_symbols = 8
    fft_size = 20
    subcarrier_spacing = 15e3
    num_tx = 2
    num_streams_per_tx = 2
    cyclic_prefix_length = 10
    
    rg = sionna.ofdm.ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                      fft_size=fft_size,
                      subcarrier_spacing=subcarrier_spacing,
                      num_tx=num_tx,
                      num_streams_per_tx=num_streams_per_tx,
                      cyclic_prefix_length=cyclic_prefix_length,
                      pilot_pattern="kronecker",
                      pilot_ofdm_symbol_indices = [2],
                      dc_null=True,
                      num_guard_carriers=(0,3))
    rg.pilot_pattern = PilotPatternWithSilence(rg.pilot_pattern, [0, 1])
    # rg.show()
    # rg.pilot_pattern.show()
    
    mash = Mash(rg)
    demash = DeMash(mash)
    num_bits = rg.num_data_symbols * 2
    # batch_size, num_tx, num_streams_per_tx, num_bits
    mapper = sionna.mapping.Mapper('qam', 2)
    rg_mapper = sionna.ofdm.ResourceGridMapper(rg)
    
    x = tf.zeros((1, num_tx, num_streams_per_tx, num_bits), dtype=tf.complex64)
    x = mapper(x)
    x = rg_mapper(x)
    x = mash(x)
    print(x.shape)
    import matplotlib.pyplot as plt
    for i in range(num_tx):
        for j in range(num_streams_per_tx):
            plt.figure()
            plt.imshow(tf.math.real(x[0,i,j]))
            plt.title(f"tx {i} stream {j}")

test()


# %%

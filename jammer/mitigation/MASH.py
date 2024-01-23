"""Mitigation via Subspace Hiding (MASH)"""
#%%
import tensorflow as tf
tf.config.run_functions_eagerly(True)
import numpy as np
import sionna
from sionna.utils import flatten_last_dims
from jammer.custom_pilots import PilotPatternWithSilence

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
        # TODO get the efficient code to work, then remove this
        self._f = self._norm_factor * tf.signal.fft(tf.eye(n, dtype=dtype))
        self._f_h = tf.linalg.adjoint(self._f)

    def multiply_from_right(self, x, adjoint=False):
        """Returns x @ A or x @ A^H."""
        if adjoint:
            for i in range(self._num_iterations-1, -1, -1):
                d = tf.linalg.diag(self._diag_values[i])
                x = x @ d
                x = x @ self._f_h
        else:
            for i in range(self._num_iterations):
                d = tf.linalg.diag(self._diag_values[i])
                x = x @ self._f
                x = x @ d
        return x
        # TODO get the efficient code below to work
        # if adjoint:
        #     # x @ A^H = (A @ x^H)^H
        #     x = tf.linalg.adjoint(x)
        #     for i in range(self._num_iterations-1, -1, -1):
        #         x = tf.linalg.diag(self._diag_values[i]) @ x
        #         x = self._norm_factor * tf.signal.fft(x)
        # else:
        #    # x @ A = (A^H @ x^H)^H
        #     x = tf.linalg.adjoint(x)
        #     for i in range(self._num_iterations):
        #         x = 1./self._norm_factor * tf.signal.ifft(x)
        #         x = tf.linalg.diag(self._diag_values[i]) @ x
        # return tf.linalg.adjoint(x)

class Mash(tf.keras.layers.Layer):
    """Intended to be used after RG-Mapper."""
    def __init__(self, resource_grid, renew_secret=True, **kwargs):
        super().__init__(**kwargs)
        self._rg = resource_grid
        self._renew_secret = renew_secret
        self._sc_ind = tf.cast(resource_grid.effective_subcarrier_ind, tf.int64)
        self.C = None

    def call(self, inputs):
        """Input:
        [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex: The whole RG."""
        # remove guard and DC bands -> [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers]
        x = tf.gather(inputs, self._sc_ind, axis=-1)
        # flatten RG
        shape_before_flatten = tf.shape(x)
        x = flatten_last_dims(x, 2)
        n = x.shape[-1]
        if self._renew_secret or self.C is None:
            self.C = HaarApproximation(n)
        # TODO do we have to flatten num_tx and num_streams_per_tx? (I think not)
        # x = sionna.utils.flatten_dims(x, 2, 1)
        # x = tf.expand_dims(x, -2)
        # x contains zeros at silent pilot indices. I.e. we can just multiply with C (and C^H) to demash.
        x = self.C.multiply_from_right(x)
        # reshape to RG without guard and DC bands
        x = tf.reshape(x, shape_before_flatten)
        # add guard and DC bands again
        x = tf.transpose(x, (4, 0, 1, 2, 3))
        indices = tf.expand_dims(self._sc_ind, -1)
        # input_shape = tf.shape(inputs, )
        x_shape = tf.concat([[tf.shape(inputs, tf.int64)[-1]], tf.shape(inputs, tf.int64)[:-1]], axis=0)
        x = tf.scatter_nd(indices, x, x_shape)
        return tf.transpose(x, (1, 2, 3, 4, 0))
        

class DeMash(tf.keras.layers.Layer):
    def __init__(self, mash, **kwargs):
        super().__init__(**kwargs)
        self._mash = mash

    def call(self, inputs):
        """Input:
        [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex: The whole RG."""
        # remove guard and DC bands -> [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, num_effective_subcarriers]
        y = tf.gather(inputs, self._mash._sc_ind, axis=-1)
        # flatten RG
        shape_before_flatten = tf.shape(y)
        y = flatten_last_dims(y, 2)
        # x contains zeros at silent pilot indices. I.e. we can just multiply with C (and C^H) to demash.
        y = self._mash.C.multiply_from_right(y, adjoint=True)
        y = tf.reshape(y, shape_before_flatten)
        # add guard and DC bands again
        y = tf.transpose(y, (4, 0, 1, 2, 3))
        indices = tf.expand_dims(self._mash._sc_ind, -1)
        x_shape = tf.concat([[tf.shape(inputs, tf.int64)[-1]], tf.shape(inputs, tf.int64)[:-1]], axis=0)
        y = tf.scatter_nd(
            indices, y, x_shape)
        # TODO should we add the real received guard and DC bands?
        return tf.transpose(y, (1, 2, 3, 4, 0))

        

def test():
    import matplotlib.pyplot as plt
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
    stream_management = sionna.mimo.StreamManagement(np.ones([1, num_tx]), num_streams_per_tx)
    lmmse_eq = sionna.ofdm.LMMSEEqualizer(rg, stream_management, whiten_interference=True)
    remove_nulled_subcarriers = sionna.ofdm.RemoveNulledSubcarriers(rg)
    
    # x = tf.zeros((1, num_tx, num_streams_per_tx, num_bits), dtype=tf.complex64)
    x = tf.random.uniform((1, num_tx, num_streams_per_tx, num_bits))
    x = tf.cast(x > 0.5, tf.complex64)
    x = mapper(x)
    x_rg = rg_mapper(x)
    x = mash(x_rg)
    x_demashed = demash(x)
    # now test through channel
    ofdm_channel = create_ofdm_channel(rg, num_tx, num_streams_per_tx)
    ebno_db = 100
    no = sionna.utils.ebnodb2no(ebno_db, 2, coderate=1.0, resource_grid=None)
    y, h = ofdm_channel([x, no])
    y_demashed = demash(y)
    h_hat = remove_nulled_subcarriers(h)
    err_var = 0.0
    x_hat, no_eff = lmmse_eq([y, h_hat, err_var, no])
    x_hat_rg = rg_mapper(x_hat)
    
    for i in range(num_tx):
        for j in range(num_streams_per_tx):
            fig, axs = plt.subplots(1, 4)
            plt.title(f"tx {i} stream {j}")
            axs[0].imshow(tf.math.real(x_rg[0,i,j]))
            axs[1].imshow(tf.math.real(x[0,i,j]))
            axs[2].imshow(tf.math.real(x_demashed[0,i,j]))
            # TODO this is receive vector
            axs[3].imshow(tf.math.real(x_hat_rg[0,i,j]))

def create_ofdm_channel(rg, num_tx, num_tx_ant, num_bs_ant=18, carrier_frequency=3.5e9, batch_size=1):
    # Configure antenna arrays
    ut_array = sionna.channel.tr38901.AntennaArray(
                        num_rows=1,
                        num_cols=num_tx_ant,
                        polarization="single",
                        polarization_type="V",
                        antenna_pattern="omni",
                        carrier_frequency=carrier_frequency)

    bs_array = sionna.channel.tr38901.AntennaArray(num_rows=1,
                        num_cols=int(num_bs_ant/2),
                        polarization="dual",
                        polarization_type="cross",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency)

    channel_parameters = {
        "carrier_frequency": carrier_frequency,
        "ut_array": ut_array,
        "bs_array": bs_array,
        "direction": "uplink",
        "enable_pathloss": False,
        "enable_shadow_fading": False,
    }
    channel_parameters["o2i_model"] = "low"
    channel = sionna.channel.tr38901.UMi(**channel_parameters)
    topology = sionna.channel.gen_single_sector_topology(batch_size,
                                          num_tx,
                                          "umi",
                                          min_ut_velocity=0.0,
                                          max_ut_velocity=0.0,
                                          indoor_probability=0.7)
    channel.set_topology(*topology)
    ofdm_channel = sionna.channel.OFDMChannel(channel, rg, add_awgn=True,
                               normalize_channel=True, return_channel=True)
    return ofdm_channel

test()
# h = HaarApproximation(3)
# x = tf.eye(3, dtype=tf.complex64)
# x = h.multiply_from_right(x)
# print(tf.norm(x, axis=1))

# %%

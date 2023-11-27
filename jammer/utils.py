#%%
import tensorflow as tf
import numpy as np
import sionna
import copy
import math
import io
import matplotlib.pyplot as plt

def sample_function(sampler, dtype):
    """
        Returns function which samples from a constellation or a distribution.
        sampler: String in ["uniform", "gaussian"], a constellation, or function with signature (shape, dtype) -> tf.Tensor, where elementwise E[|x|^2] = 1
    """
    if isinstance(sampler, str):
        if sampler == "uniform":
            sample_function = sample_complex_uniform_disk
        elif sampler == "gaussian":
            sample_function = sample_complex_gaussian
        else:
            raise ValueError(f"Unknown sampler {sampler}")
    elif isinstance(sampler, sionna.mapping.Constellation):
        sample_function = constellation_to_sampler(sampler, dtype=dtype)
    elif callable(sampler):
        sample_function = sampler
    else:
        raise ValueError(f"Unknown sampler {sampler}")
    return sample_function


def sample_complex_uniform_disk(shape, dtype):
    """Sample from complex plane with E[|x|^2] = 1]. In this case, we sample from uniform circle.
    Sample theta and R uniform, r = sqrt(2R)"""
    r = tf.complex(tf.random.uniform(shape, minval=0, maxval=1, dtype=dtype.real_dtype), tf.cast(0.0, dtype.real_dtype))
    theta = tf.complex(tf.random.uniform(shape, minval=0, maxval=2*np.pi, dtype=dtype.real_dtype), tf.cast(0.0, dtype.real_dtype))
    return tf.sqrt(2*r)*tf.exp(1j*theta)

def sample_complex_gaussian(shape, dtype):
    """Sample from complex plane with E[|x|^2] = 1]. In this case, we sample from a complex gaussian."""
    stddev = np.sqrt(0.5)
    return tf.complex(tf.random.normal(shape, stddev=stddev, dtype=dtype.real_dtype), tf.random.normal(shape, stddev=stddev, dtype=dtype.real_dtype))

def constellation_to_sampler(constellation, dtype):
    """Convert a constellation to a sampler. We normalize the constellation so that it is similar to other distributions."""
    binary_source = sionna.utils.BinarySource()
    _constellation = copy.copy(constellation)
    _constellation.normalize = True
    mapper = sionna.mapping.Mapper(constellation=_constellation, dtype=dtype)
    def sampler(shape, dtype):
        """Sample from a constellation"""
        assert dtype == mapper.dtype
        binary_source_shape = shape[:-1] + [shape[-1] * _constellation.num_bits_per_symbol]
        bits = binary_source(binary_source_shape)
        return mapper(bits)
        
    return sampler


def covariance_estimation_from_signals(y, num_odfm_symbols):
    """Estimate the covariance matrix of a signal y.
    y: [batch_size, num_rx, num_rx_ant, num_jammer_symbols, fft_size], tf.complex
        where num_jammer_symbols is the number of symbols where only the jammer is sending
    num_ofdm_symbols: int
    output: [batch_size, num_rx, num_ofdm_symbols, fft_size, num_rx_ant, num_rx_ant]
    """
    # we calculate the covariance matrix, broadcast it over all ofdm symbols
    y = tf.transpose(y, perm=[0, 1, 4, 2, 3]) # [batch_size, num_rx, fft_size, num_rx_ant, num_jammer_symbols]
    # 1/N * y*y^H
    cov = tf.matmul(y, y, adjoint_b=True)/tf.cast(tf.shape(y)[-1], y.dtype)
    # add num_ofdm_symbols dimension by repeating
    return tf.tile(cov[:, :, tf.newaxis, :, :, :], [1, 1, num_odfm_symbols, 1, 1, 1])

    
def ofdm_frequency_response_from_cir(a, tau, rg, normalize):
    """Calulate the frequency response of a channel from its CIR. Does this by downsampling channel gains (a) and computing DFT.
    normalize: bool. If true, normalizes over one resource grid."""
    frequencies = sionna.channel.subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
    a_freq = a[...,rg.cyclic_prefix_length:-1:(rg.fft_size+rg.cyclic_prefix_length)]
    a_freq = a_freq[...,:rg.num_ofdm_symbols]
    h_freq = sionna.channel.cir_to_ofdm_channel(frequencies, a_freq, tau, normalize=normalize)
    return h_freq



# TODO possibly make this use sparse tensors for better performance
# TODO should we integrate this more general function into jammer (let every input dimension have the option to be sparse)
def sparse_mask(shape, sparsity, dtype=tf.float64):
    """shape: list of int. Output shape
    sparsity: list of float. Same length as shape. Probability of slice being one.
    Returns a mask with 1.0 for non-zero elements and 0.0 for zero elements."""
    assert len(shape) == len(sparsity)

    mask = np.ones(shape, dtype=dtype.as_numpy_dtype)
    for i, s in enumerate(shape):
        assert s > 0
        zero_indices = np.random.choice(s, size=round(s*(1-sparsity[i])), replace=False)
        mask_index = [slice(None)] * len(shape)
        mask_index[i] = zero_indices
        mask[tuple(mask_index)] = 0.0

    return tf.convert_to_tensor(mask)

def reduce_matrix_rank(matrix, rank):
    """Reduce the rank of a matrix by setting the smallest singular values to zero.
    matrix: [..., M, N]
    rank: int. Desired rank of matrix.
    """
    s, u, v = tf.linalg.svd(matrix, full_matrices=False, compute_uv=True)
    s = tf.cast(s, matrix.dtype)
    # set smallest singular values to zero
    s = tf.where(tf.range(tf.shape(s)[-1]) < rank, s, tf.zeros_like(s, dtype=matrix.dtype))
    # reconstruct matrix
    return tf.matmul(tf.matmul(u, tf.linalg.diag(s)), v, adjoint_b=True)

def db_to_linear(db):
    """Converts from dB to linear scale."""
    return 10.0**(db/10.0)

def linear_to_db(linear):
    """Converts from linear to dB scale."""
    return 10*math.log(float(linear))/math.log(10.0)

class NonNegMaxMeanSquareNorm(tf.keras.constraints.Constraint):
    """Ensures nonnegativity and that the mean of the squared norm of the weights is at most max_squared_norm."""
    def __init__(self, max_mean_squared_norm=1.0, axis=None):
        self.max_mean_squared_norm = max_mean_squared_norm
        self.axis = axis

    def __call__(self, w):
        w_nonneg = tf.maximum(w, 0.0)
        mean_squared_norm = tf.reduce_mean(tf.square(w_nonneg), axis=self.axis, keepdims=True)
        if(mean_squared_norm > self.max_mean_squared_norm):
            scale = tf.sqrt(self.max_mean_squared_norm / (mean_squared_norm + tf.keras.backend.epsilon()))
            return w_nonneg * scale
        else:
            return w_nonneg

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

def plot_matrix(a, figsize=(6.4, 4.8)):
    """Plots a matrix as a heatmap. Works only in eager mode."""
    fig, ax = plt.subplots(figsize=figsize)
    # im = ax.imshow(a)
    im = ax.matshow(a, aspect='auto')
    fig.colorbar(im, fraction=0.05)
    return fig

def matrix_to_image(a):
    """Converts a matrix to an image. Works only in eager mode."""
    fig = plot_matrix(a, figsize=(6.0, 2.4))
    return plot_to_image(fig)

def expected_bitflips(y_true, y_pred, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE):
    """Expected BER loss. y_true in {0, 1}, y_pred in [0, 1]."""
    losses = tf.reduce_sum(y_true * (1-y_pred) + (1-y_true) * (y_pred), axis=-1)
    # reduce over batch dimension, according to parameter reduction <-TODO
    return tf.reduce_mean(losses)
    

# y_true = tf.constant([[0.0, 1.0],
#                       [1.0, 0.0]])
# y_pred = tf.constant([[0.1, 0.9],
#                       [0.0, 1.0]])
# print(expected_bitflips(y_true, y_pred))

plot_matrix(tf.random.normal([14, 128]), figsize=(6.0, 2.4))
# %%

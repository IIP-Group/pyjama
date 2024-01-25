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

    Input
    -----
        sampler : str | Constellation | callable
            String in `["uniform", "gaussian"]`, an instance of :class:`~sionna.mapping.Constellation`, or function with signature ``(shape, dtype) -> tf.Tensor``,
            where elementwise :math:`E[|x|^2] = 1`.
        dtype : tf.Dtype
            Defines the datatype the returned function should return.

    Output
    ------
        callable
            Function with signature ``(shape, dtype) -> tf.Tensor`` which returns a tensor of shape ``shape`` with dtype ``dtype``.
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
    """
    Sample uniform circle on complex plane.

    Input
    -----
        shape : list of int
            Shape of tensor to return.
        dtype : tf.complex
            Datatype of tensor to return.

    Output
    ------
        : [shape], ``dtype``
            Each element is sampled from within a circle with radius :math:`\sqrt{2}`.
            This results in element-wise :math:`E[|x|^2] = 1`.
    """
    # Sample theta and R uniform, r = sqrt(2R)
    r = tf.complex(tf.random.uniform(shape, minval=0, maxval=1, dtype=dtype.real_dtype), tf.cast(0.0, dtype.real_dtype))
    theta = tf.complex(tf.random.uniform(shape, minval=0, maxval=2*np.pi, dtype=dtype.real_dtype), tf.cast(0.0, dtype.real_dtype))
    return tf.sqrt(2*r)*tf.exp(1j*theta)

def sample_complex_gaussian(shape, dtype):
    """
    Sample complex gaussian.
    
    Input
    -----
        shape : list of int
            Shape of tensor to return.
        dtype : tf.complex
            Datatype of tensor to return.
        
    Output
    ------
        : [shape], ``dtype``
            Each element is sampled from a complex gaussian with variance 1/2.
            This results in element-wise :math:`E[|x|^2] = 1`.
    """
    stddev = np.sqrt(0.5)
    return tf.complex(tf.random.normal(shape, stddev=stddev, dtype=dtype.real_dtype), tf.random.normal(shape, stddev=stddev, dtype=dtype.real_dtype))

def constellation_to_sampler(constellation, normalize=True, dtype=tf.complex64):
    """
    Convert a constellation to a function which samples the constellation.

    Input
    -----
        constellation : Constellation
            An instance of :class:`~sionna.mapping.Constellation` to sample from.
        normalize : bool
            If True, normalize the constellation so that the average power of each symbol is 1.
            
    Output
    ------
        : callable, ``(shape, dtype) -> tf.Tensor``
            Function which samples the constellation.
    """
    binary_source = sionna.utils.BinarySource()
    if normalize:
        _constellation = copy.copy(constellation)
        _constellation.normalize = True
    else:
        _constellation = constellation
    mapper = sionna.mapping.Mapper(constellation=_constellation, dtype=dtype)
    def sampler(shape, dtype):
        """Sample from a constellation"""
        assert dtype == mapper.dtype
        binary_source_shape = shape[:-1] + [shape[-1] * _constellation.num_bits_per_symbol]
        bits = binary_source(binary_source_shape)
        return mapper(bits)
        
    return sampler


def covariance_estimation_from_signals(y, num_odfm_symbols):
    """
    Estimate the covariance matrix of a signal y.
    
    Input
    -----
    y : [batch_size, num_rx, num_rx_ant, num_symbols, fft_size], tf.complex
        ``num_symbols`` is the number of symbols over which we estimate the covariance matrix
        (e.g. the number of symbols where only a jammer is transmitting).
    num_ofdm_symbols: int
        Number of OFDM symbols in the complete resource grid.

    Output
    ------
    : [batch_size, num_rx, num_ofdm_symbols, fft_size, num_rx_ant, num_rx_ant], y.dtype
        Covariance matrix over rx antennas, for each batch, rx and subcarrier. Broadcasted over ``num_ofdm_symbols``.
    """
    # we calculate the covariance matrix, broadcast it over all ofdm symbols
    y = tf.transpose(y, perm=[0, 1, 4, 2, 3]) # [batch_size, num_rx, fft_size, num_rx_ant, num_jammer_symbols]
    # 1/N * y*y^H
    cov = tf.matmul(y, y, adjoint_b=True)/tf.cast(tf.shape(y)[-1], y.dtype)
    # add num_ofdm_symbols dimension by repeating
    return tf.tile(cov[:, :, tf.newaxis, :, :, :], [1, 1, num_odfm_symbols, 1, 1, 1])

# def ofdm_frequency_response_from_cir(a, tau, rg, normalize):
#     """Calulate the frequency response of a channel from its CIR. Does this by downsampling channel gains (a) and computing DFT.
#     normalize: bool. If true, normalizes over one resource grid."""
#     frequencies = sionna.channel.subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
#     a_freq = a[...,rg.cyclic_prefix_length:-1:(rg.fft_size+rg.cyclic_prefix_length)]
#     a_freq = a_freq[...,:rg.num_ofdm_symbols]
#     h_freq = sionna.channel.cir_to_ofdm_channel(frequencies, a_freq, tau, normalize=normalize)
#     return h_freq


# TODO possibly make this use sparse tensors for better performance
# TODO should we integrate this more general function into jammer (let every input dimension have the option to be sparse)
# def sparse_mask(shape, sparsity, dtype=tf.float64):
#     """shape: list of int. Output shape
#     sparsity: list of float. Same length as shape. Probability of slice being one.
#     Returns a mask with 1.0 for non-zero elements and 0.0 for zero elements."""
#     assert len(shape) == len(sparsity)

#     mask = np.ones(shape, dtype=dtype.as_numpy_dtype)
#     for i, s in enumerate(shape):
#         assert s > 0
#         zero_indices = np.random.choice(s, size=round(s*(1-sparsity[i])), replace=False)
#         mask_index = [slice(None)] * len(shape)
#         mask_index[i] = zero_indices
#         mask[tuple(mask_index)] = 0.0

#     return tf.convert_to_tensor(mask)

def reduce_matrix_rank(matrix, rank):
    """
    Reduce the rank of a matrix by setting the smallest singular values to zero.

    Input
    -----
    matrix: [..., M, N]
    rank: int.
        Desired rank of matrix.
    
    Output
    ------
    : [..., M, N]
        Matrix with rank smaller or equal ``rank``.
    """
    s, u, v = tf.linalg.svd(matrix, full_matrices=False, compute_uv=True)
    s = tf.cast(s, matrix.dtype)
    # set smallest singular values to zero
    s = tf.where(tf.range(tf.shape(s)[-1]) < rank, s, tf.zeros_like(s, dtype=matrix.dtype))
    # reconstruct matrix
    return tf.matmul(tf.matmul(u, tf.linalg.diag(s)), v, adjoint_b=True)

def db_to_linear(db):
    """
    Converts number from dB to linear scale.

    Input
    -----
    db : float
        Number in dB.

    Output
    ------
    : float
        :math:`10^{db/10}`
    """
    return 10.0**(db/10.0)

def linear_to_db(linear):
    """
    Converts number from linear to dB scale.
    
    Input
    -----
    linear : float
        Number from linear scale.

    Output
    ------
    : float
        :math:`10\log_{10}(linear)`
    """
    return 10*math.log(float(linear))/math.log(10.0)

class NonNegMaxMeanSquareNorm(tf.keras.constraints.Constraint):
    r"""
    Scales the input tensor so that the mean power of each element along the given axis is at most ``max_mean_squared_norm``.
    Also ensures that all elements are non-negative.

    Ensures that :math:`\frac{1}{n} \sum{|w_i|^2} \le \mathtt{max\_mean\_squared\_norm}` along ``axis``
    and that :math:`w_i \ge 0` for all values in ``w``.
    ``n`` is the number of elements in ``w`` along the axis ``axis``.
    
    Parameters
    ----------
    max_mean_squared_norm : float
        Maximum to which all elements should be scaled, so that the mean squared norm does not exceed this value.
    axis : int or list of int
        Axis along which the mean squared norm is calculated.
        
    Input
    -----
    w : tf.Tensor
        Tensor to which the constraint is applied.

    Output
    ------
    : Same shape as ``w``, ``w.dtype``
        If the constraint is valid, ``w`` is returned unchanged. Otherwise, ``w`` is scaled so that the constraint is valid.
        All values in ``w`` which were negative are set to zero.
    """
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

class MaxMeanSquareNorm(tf.keras.constraints.Constraint):
    r"""
    Scales the input tensor so that the mean power of each element along the given axis is at most ``max_mean_squared_norm``.

    Ensures that :math:`\frac{1}{n} \sum{|w|^2} \le \mathtt{max\_mean\_squared\_norm}`.
    ``n`` is the number of elements in ``w`` along the axis ``axis``.
    
    Parameters
    ----------
    max_mean_squared_norm : float
        Maximum to which all elements should be scaled, so that the mean squared norm does not exceed this value.
    axis : int or list of int
        Axis along which the mean squared norm is calculated.
        
    Input
    -----
    w : tf.Tensor
        Tensor to which the constraint is applied.

    Output
    ------
    : Same shape as ``w``, ``w.dtype``
        If the constraint is valid, ``w`` is returned unchanged. Otherwise, ``w`` is scaled so that the constraint is valid.
    """
    def __init__(self, max_mean_squared_norm=1.0, axis=None):
        self.max_mean_squared_norm = max_mean_squared_norm
        self.axis = axis

    def __call__(self, w):
        mean_squared_norm = tf.reduce_mean(tf.square(tf.abs(w)), axis=self.axis, keepdims=True)
        if(mean_squared_norm > self.max_mean_squared_norm):
            scale = tf.sqrt(self.max_mean_squared_norm / (mean_squared_norm + tf.keras.backend.epsilon()))
            return w * scale
        else:
            return w

def reduce_mean_power(a, axis=None, keepdims=False):
    """
    Calculates the mean power of a tensor along the given axis.
    
    Input
    -----
    a : tf.Tensor
        Tensor of which the mean power is calculated.
    axis : int or list of int
        Axis along which the mean power is calculated. If None, the mean power is calculated over all axes.
    keepdims : bool
        If True, the reduced dimensions are kept with size 1.

    Output
    ------
    : tf.Tensor
        Contains the mean power of ``a``, calculated along ``axis``.
    """
    return tf.reduce_mean(tf.square(tf.abs(a)), axis=axis, keepdims=keepdims)

def normalize_power(a, is_amplitude=True):
    """
    Scales input tensor so that the mean power per element is 1.

    Input
    -----
    a : tf.Tensor
        Tensor to be normalized.
    is_amplitude : bool
        If True, ``a`` is assumed to be the amplitude, otherwise it is assumed to be the power.
    
    Output
    ------
    : tf.Tensor
        Tensor with mean power of 1. It can be interpreted as amplitude or power like the input.
    """
    if is_amplitude:
        return a / tf.sqrt(reduce_mean_power(a))
    else:
        return a / tf.reduce_mean(a)


def plot_to_image(figure):
    """
    Converts a matplotlib figure to a PNG image and returns it. The supplied figure is closed and inaccessible after this call.
    
    Input
    -----
    figure: Figure
        An instance of :class:`~matplotlib.figure.Figure` to convert to an image tensor.

    Output
    ------
    : [1, height, width, 4], tf.uint8
        Image of the figure.
    """
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
    """
    Plots a matrix as a heatmap.
    If `a` has more than 2 dimensions, ``a[0, 0 .., :, :]`` is plotted.
    
    Input
    -----
    a : [..., M, N]
        Matrix to plot.
    figsize : (float, float)
        width, height in inches
    
    """
    if len(a.shape) > 2:
        a = a[(0,) * (len(a.shape) - 2)]
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.matshow(a, aspect='auto')
    fig.colorbar(im, fraction=0.05)
    return fig

def matrix_to_image(a):
    """Converts a matrix to an image."""
    fig = plot_matrix(a, figsize=(6.0, 2.4))
    return plot_to_image(fig)

# TODO: this is just L1 loss, isn't it?
# def expected_bitflips(y_true, y_pred, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE):
#     """Expected BER loss. y_true in {0, 1}, y_pred in [0, 1]."""
#     losses = tf.reduce_sum(y_true * (1-y_pred) + (1-y_true) * (y_pred), axis=-1)
#     # reduce over batch dimension, according to parameter reduction <-TODO
#     return tf.reduce_mean(losses)
    
# TODO test with alpha decreasing through training (and exp. scaling) (i.e. more weight on last iteration as training progresses)
class IterationLoss(tf.keras.losses.Loss):
    r"""
    Loss function for iterative decoder which returns the output of each decoder iteration.
    
    Calculates the loss for each iteration separately and returns the weighted sum.
    If ``exponential_alpha_scaling`` is true, the loss for each iteration is :math:`\sum_i \alpha^{n-i} * |b - \hat{b}_i|`,
    where :math:`i` is the iteration number and :math:`n` the total number of iterations.
    Otherwise, each iteration loss is scaled by :math:`\alpha` instead.
    
    Parameters
    ----------
    alpha : float
        Weights applied
    exponential_alpha_scaling : bool
        If true, later iterations are weighted more heavily. Otherwise, all iterations are weighted equally.
    reduction : Reduction
        A :class:`~tf.keras.losses.Reduction` to apply to the loss.

    Input
    -----
    b : [batch_size, num_bits]
        Ground truth bits.
    b_hat_iterations : [batch_size, num_bits * num_iterations], float
        Output of decoder iterations.

    Output
    ------
    : float
        Loss value.
    """
    # TODO implement reduction
    def __init__(self, alpha=0.5, exponential_alpha_scaling=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE):
        self._alpha = alpha
        self._exponention_alpha_scaling = exponential_alpha_scaling
        super(IterationLoss, self).__init__(reduction=reduction)

    def call(self, b, b_hat_iterations):
        # batch, num_bits * num_iterations -> batch, num_bits, num_iterations
        b_hat_iterations = tf.reshape(b_hat_iterations, [b_hat_iterations.shape[0], b.shape[1], -1])

        num_iterations = b_hat_iterations.shape[2]
        alpha = tf.ones([num_iterations]) * self._alpha
        if self._exponention_alpha_scaling:
            alpha = alpha ** tf.range(num_iterations-1, -0.1, -1, dtype=alpha.dtype)

        difference = tf.abs(b[..., tf.newaxis] - b_hat_iterations)
        difference *= alpha
        return tf.reduce_mean(tf.reduce_sum(difference, axis=2))
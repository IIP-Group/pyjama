import tensorflow as tf
import numpy as np
import sionna
import copy

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
        sample_function = constellation_to_sampler(sampler, dtype=self._dtype_as_dtype)
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
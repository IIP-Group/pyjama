import tensorflow as tf
import numpy as np
import sionna
import copy

def _sample_complex_uniform_disk(shape, dtype):
    """Sample from complex plane with E[|x|^2] = 1]. In this case, we sample from uniform circle.
    Sample theta and R uniform, r = sqrt(2R)"""
    r = tf.complex(tf.random.uniform(shape, minval=0, maxval=1, dtype=dtype.real_dtype), tf.cast(0.0, dtype.real_dtype))
    theta = tf.complex(tf.random.uniform(shape, minval=0, maxval=2*np.pi, dtype=dtype.real_dtype), tf.cast(0.0, dtype.real_dtype))
    return tf.sqrt(2*r)*tf.exp(1j*theta)

def _sample_complex_gaussian(shape, dtype):
    """Sample from complex plane with E[|x|^2] = 1]. In this case, we sample from a complex gaussian."""
    stddev = np.sqrt(0.5)
    return tf.complex(tf.random.normal(shape, stddev=stddev, dtype=dtype.real_dtype), tf.random.normal(shape, stddev=stddev, dtype=dtype.real_dtype))

def _constellation_to_sampler(constellation, dtype):
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


def covariance_estimation_from_signals(y):
    """Estimate the covariance matrix of a signal y.
    y: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex
    output: [batch_size, num_rx, num_ofdm_symbols, fft_size, num_rx_ant, num_rx_ant]
    """
    y = tf.transpose(y, perm=[0, 1, 3, 4, 2]) # [batch_size, num_rx, num_ofdm_symbols, fft_size, rx_ant]
    return tf.matmul(y, y, adjoint_a=True)/tf.cast(tf.shape(y)[-1], y.dtype)
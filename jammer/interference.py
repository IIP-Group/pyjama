import sionna
import tensorflow as tf
from sionna.ofdm import LinearInterpolator

class CovarianceEstimator(tf.keras.Layer):
    def __init__(pilot_pattern):
        super(CovarianceEstimator, self).__init__()
        self._pilot_pattern = pilot_pattern
        # TODO don't do it with sionnas interpolators:
        # abuse of signature & inefficient  (over whole RG)
        # self._interpolator = LinearInterpolator()

    def call(self, inputs):
        
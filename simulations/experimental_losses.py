import tensorflow as tf

class QuadraticCutoffLoss():
    ''' Function with input domain [0, 1], and output domain [0, 1].
        f(x) = 0 for x = 0
        f'(x) = x_prime_0 for x = 0
        f(x) = a*x^2 + bx for 0 < x < x_cutoff (for some a, b)
        f(x) = 1 for x >= x_cutoff
    '''
    
    def __init__(self, x_cutoff=0.6, f_prime_0=0.1):
        self._x_cutoff = x_cutoff
        self._x_prime_0 = f_prime_0
        a = 1/(x_cutoff*x_cutoff) - f_prime_0/x_cutoff
        b = f_prime_0
        self.coeffs = [a, b, 0.0]
        
    def __call__(self, y_true, y_pred):
        x = tf.math.abs(y_true - y_pred)
        results =  tf.where(x < self._x_cutoff, tf.math.polyval(self.coeffs, x), tf.constant(1.0))
        # TODO: parameterize reduction?
        return tf.reduce_mean(results)
# %%
import tensorflow as tf
import numpy as np
import sionna
from sionna.ofdm import PilotPattern, KroneckerPilotPattern

class OneHotWithSilencePilotPattern(PilotPattern):
    """Creates one-hot pilot pattern. Per stream one OFDM symbol.
    After the pilots, silence is transmitted.

    Parameters
    ----------
    num_tx : int
        Number of transmitters.

    num_streams_per_tx : int
        Number of streams per transmitter.

    num_ofdm_symbols : int
        Number of OFDM symbols.

    num_effective_subcarriers : int
        Number of effective subcarriers
        that are available for the transmission of data and pilots.
        Note that this number is generally smaller than the ``fft_size``
        due to nulled subcarriers.

    num_silence_symbols : int
        Number of silence symbols before the pilots.

    dtype : tf.Dtype
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `tf.complex64`.
    """
    def __init__(self,
                 num_tx,
                 num_streams_per_tx,
                 num_ofdm_symbols,
                 num_effective_subcarriers,
                 num_silence_symbols=0,
                 dtype=tf.complex64):

        assert num_tx > 0, \
            "`num_tx` must be positive`."
        assert num_streams_per_tx > 0, \
            "`num_streams_per_tx` must be positive`."
        assert num_ofdm_symbols > 0, \
            "`num_ofdm_symbols` must be positive`."
        assert num_effective_subcarriers > 0, \
            "`num_effective_subcarriers` must be positive`."
        assert num_silence_symbols >= 0, \
            "`num_silence_symbols` must be positive or zero`."

        shape = [num_tx, num_streams_per_tx, num_ofdm_symbols,
                      num_effective_subcarriers]
        num_streams = num_streams_per_tx * num_tx
        num_pilots = num_streams + num_silence_symbols
        mask = np.zeros(shape, dtype=bool)
        mask[:, :, :num_pilots, :] = True
        # we just use symbols 1+0j as pilots, but could also e.g. sample from constellation
        pilots = tf.concat(
            [tf.zeros(shape[:2] + [num_silence_symbols * num_effective_subcarriers], dtype=dtype),
             tf.ones(shape[:2] + [num_streams * num_effective_subcarriers], dtype=dtype)], axis=-1)
        print(pilots.shape)
        #tf.zeros(shape[:2] + [num_], dtype=dtype)

        # empty pilots are modeled as zero symbols (masks are still 1 at these positions)
        # this is compatiple with OFDM channel and ResourceGrid(De)Mapper.
        # TODO check if compatible with channel estimators. It is at least with the LS estimator.
        # TODO Interpolators are a possible problem. Check all interpolators.
        # TODO: check 5G NR standard for pilot pattern with silence. There, they also use zero symbols.
        super().__init__(mask, pilots, trainable=False, normalize=False,
                         dtype=dtype)

# pp = OneHotWithSilencePilotPattern(3, 1, 14, 12, 2)
# pp.show(show_pilot_ind=True)

# %%

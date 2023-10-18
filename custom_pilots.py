# %%
import tensorflow as tf
import numpy as np
import sionna
from sionna.ofdm import PilotPattern, KroneckerPilotPattern

# TODO write classe WithSilencePilotPattern which takes other PilotPattern as input and adds silence at specified positions. Might have silence_ind property. Use tf.concat
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
        assert num_ofdm_symbols > num_silence_symbols + num_tx * num_streams_per_tx, \
            "`num_ofdm_symbols` must be greater than `num_silence_symbols + num_tx * num_streams_per_tx`."

        shape = [num_tx, num_streams_per_tx, num_ofdm_symbols,
                      num_effective_subcarriers]
        num_streams = num_streams_per_tx * num_tx
        num_pilots = num_streams + num_silence_symbols
        mask = np.zeros(shape, dtype=bool)
        mask[:, :, :num_pilots, :] = True
        # we just use symbols 1+0j as pilots, but could also e.g. sample from constellation
        shape[2] = num_pilots
        pilots = np.zeros(shape, dtype=np.complex64)
        for i in range(num_tx):
            for j in range(num_streams_per_tx):
                pilots[i, j, num_silence_symbols + i * num_streams_per_tx + j, :] = 1+0j
        pilots = np.reshape(pilots, [num_tx, num_streams_per_tx, -1])

        # empty pilots are modeled as zero symbols (masks are still 1 at these positions)
        # this is compatiple with OFDM channel and ResourceGrid(De)Mapper.
        # TODO check if compatible with channel estimators. It is at least with the LS estimator.
        # TODO Interpolators are a possible problem. Check all interpolators. NN interpolator is fine.
        # TODO: check 5G NR standard for pilot pattern with silence. There, they also use zero symbols.
        super().__init__(mask, pilots, trainable=False, normalize=False,
                         dtype=dtype)

# pp = OneHotWithSilencePilotPattern(3, 2, 14, 7, 2)
# pp.show(show_pilot_ind=True)

# %%

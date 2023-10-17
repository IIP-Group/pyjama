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
        Number of silence symbols after the pilots.

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
        assert num_ofdm_symbols >= num_silence_symbols + num_tx*num_streams_per_tx, \
            "`num_ofdm_symbols` must be greater or equal to `num_tx`*`num_streams_per_tx`."
        assert num_silence_symbols >= 0, \
            "`num_silence_symbols` must be positive or zero`."

        shape = [num_tx, num_streams_per_tx, num_ofdm_symbols,
                      num_effective_subcarriers]
        num_streams = num_streams_per_tx * num_tx
        mask = np.zeros(shape, dtype=bool)
        mask[:, :, :num_silence_symbols + num_streams, :] = True
        pilots = tf.zeros(shape[:2] + [num_streams], dtype=dtype)

        # TODO: PilotPattern does not accept empty pilots. Find solution compatible with Channel Estimators.
        # TODO: decision: use zero symbols when quiet. Is compatible with Mappers and ODFM channel. Check if compatible with Channel Estimators.
        # TODO: check 5G NR standard for pilot pattern with silence. There, they also use zero symbols.
        super().__init__(mask, pilots, trainable=False, normalize=False,
                         dtype=dtype)

pp = OneHotWithSilencePilotPattern(3, 1, 14, 12, 2)
pp.show(show_pilot_ind=True)
sionna.nr.pusch_pilot_pattern
sionna.nr.pusch_receiver

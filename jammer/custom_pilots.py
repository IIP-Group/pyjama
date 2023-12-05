# %%
import tensorflow as tf
import numpy as np
import sionna
from sionna.ofdm import PilotPattern, KroneckerPilotPattern

# TODO this is not a very efficient implementation. Had to use python lists because numpy arrays can not temporarily have different shapes.
# could e.g. work on flattened pilot array
class PilotPatternWithSilence(PilotPattern):
    """Takes a PilotPattern and adds silence at specified OFDM symbol positions.
    Can only silence all streams and subcarriers at once.
    The passed pilot pattern may not have a pilot at any of the silence positions.
    Does not support training at the moment (silent symbols must be frozen for this).

    ATTENTION: As this adds pilots with 0 energy, but ebnobd2no expects normalized pilots,
    this class only supports unnormalized ebnodb2db (i.e. with resource_grid=None).

    Parameters
    ----------
    pilot_pattern : PilotPattern
        Pilot pattern to be used as basis.
        
    silent_ofdm_symbol_indices : list of indices (int)
    """
    def __init__(self, pilot_pattern, silent_ofdm_symbol_indices):
        self._internal_pilot_pattern = pilot_pattern
        self._silent_ofdm_symbol_indices = silent_ofdm_symbol_indices
        mask = pilot_pattern.mask.numpy().astype(bool)
        pilots = pilot_pattern.pilots.numpy().tolist()
        num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers = mask.shape
        
        # TODO check if there are no pilots at the silent positions
        
        # mask ([num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], bool)
        mask[:, :, silent_ofdm_symbol_indices, :] = True
        # pilots ([num_tx, num_streams_per_tx, num_pilots], tf.complex)
        for silent_index in silent_ofdm_symbol_indices:
            # [num_tx, num_streams_per_tx]: for each stream, how many pilots are before the silent symbol?
            num_pilots_before = np.sum(mask[:, :, :silent_index, :], axis=(2,3))
            for i in range(num_tx):
                for j in range(num_streams_per_tx):
                    insert_index = num_pilots_before[i, j] + 1
                    pilots[i][j][insert_index:insert_index] = [0+0j] * num_effective_subcarriers
                    # pilots = np.insert(pilots, num_pilots_before[i, j], 0+0j, axis=2)
                    #pilots[i, j, num_pilots_before[i, j], :] = 0+0j

        pilots = np.array(pilots)
        super().__init__(mask, pilots, trainable=False, normalize=False,
                         dtype=pilot_pattern._dtype)

        # empty pilots are modeled as zero symbols (masks are still 1 at these positions)
        # this is compatiple with OFDM channel and ResourceGrid(De)Mapper.
        # TODO check if compatible with channel estimators. It is at least with the LS estimator.
        # TODO Interpolators are a possible problem. Check all interpolators. NN interpolator is fine.
        # TODO: check 5G NR standard for pilot pattern with silence. There, they also use zero symbols.
    
    @property
    def silent_ofdm_symbol_indices(self):
        return self._silent_ofdm_symbol_indices
    

class OneHotPilotPattern(PilotPattern):
    """Creates one-hot pilot pattern. Per stream one OFDM symbol.

    Parameters
    ----------
    starting_symbol : int
        OFDM symbol index at which the pilots start.
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

    dtype : tf.Dtype
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `tf.complex64`.
    """
    def __init__(self, starting_symbol, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers, normalize=True, dtype=tf.complex64):
        assert starting_symbol >= 0, \
            "`starting_symbol` must be positive or zero`."
        assert num_tx > 0, \
            "`num_tx` must be positive`."
        assert num_streams_per_tx > 0, \
            "`num_streams_per_tx` must be positive`."
        assert num_ofdm_symbols > 0, \
            "`num_ofdm_symbols` must be positive`."
        assert num_effective_subcarriers > 0, \
            "`num_effective_subcarriers` must be positive`."
        assert num_ofdm_symbols > starting_symbol + num_tx * num_streams_per_tx, \
            "`num_ofdm_symbols` must be greater than `starting_symbol + num_tx * num_streams_per_tx`."

        shape = [num_tx, num_streams_per_tx, num_ofdm_symbols,
                      num_effective_subcarriers]
        num_streams = num_streams_per_tx * num_tx
        mask = np.zeros(shape, dtype=bool)
        mask[:, :, starting_symbol:starting_symbol+num_streams, :] = True
        # we just use symbols 1+0j as pilots, but could also e.g. sample from constellation
        shape[2] = num_streams
        pilots = np.zeros(shape, dtype=np.complex64)
        for i in range(num_tx):
            for j in range(num_streams_per_tx):
                pilots[i, j, i * num_streams_per_tx + j, :] = 1+0j
        pilots = np.reshape(pilots, [num_tx, num_streams_per_tx, -1])
        
        super().__init__(mask, pilots, trainable=False, normalize=normalize,
                         dtype=dtype)
    
    
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
                 normalize=True,
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

        super().__init__(mask, pilots, trainable=False, normalize=normalize,
                         dtype=dtype)

# pp = OneHotWithSilencePilotPattern(3, 2, 14, 7, 2)
# pp.show(show_pilot_ind=True)

# %%

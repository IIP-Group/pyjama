import numpy as np
import tensorflow as tf
from sionna.ofdm import PilotPattern

# TODO remove
# TODO attention: this is only prototype! does not work with interleaved pilots
class PilotPatternWithStutteredSilence(PilotPattern):
    """
    Takes a PilotPattern and adds silence for all transmitters, streams and subcarriers at specified OFDM symbol positions.
    The passed pilot pattern may not have a pilot at any of the silence positions.
    Does not support training at the moment (silent symbols must be frozen for this).

    ATTENTION: As this adds pilots with 0 energy, but ebnobd2no expects normalized pilots,
    this class only supports unnormalized ebnodb2db (i.e. with resource_grid=None).

    Parameters
    ----------
    pilot_pattern : PilotPattern
        Pilot pattern to be used as basis.
        
    silent_ofdm_symbol_indices : list of int
        OFDM symbol indices at which silence should be added.

    every_n_subcarriers : int
    """
    def __init__(self, pilot_pattern, silent_ofdm_symbol_indices, every_n_subcarriers=1):
        self._internal_pilot_pattern = pilot_pattern
        self._silent_ofdm_symbol_indices = silent_ofdm_symbol_indices
        self._every_n_subcarriers = every_n_subcarriers
        mask = pilot_pattern.mask.numpy().astype(bool)
        pilots = pilot_pattern.pilots.numpy().tolist()
        num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers = mask.shape
        
        # TODO check if there are no pilots at the silent positions (via assertion)
        
        # mask ([num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], bool)
        mask[:, :, silent_ofdm_symbol_indices, ::every_n_subcarriers] = True
        # pilots ([num_tx, num_streams_per_tx, num_pilots], tf.complex)
        for silent_index in silent_ofdm_symbol_indices:
            # [num_tx, num_streams_per_tx]: for each stream, how many pilots are before the silent symbol?
            num_pilots_before = np.sum(mask[:, :, :silent_index, :], axis=(2,3))
            for i in range(num_tx):
                for j in range(num_streams_per_tx):
                    insert_index = num_pilots_before[i, j]
                    num_silent_pilots_in_subcarrier = np.floor((num_effective_subcarriers - 1) / every_n_subcarriers).astype(int) + 1
                    pilots[i][j][insert_index:insert_index] = [0+0j] * num_silent_pilots_in_subcarrier

        pilots = np.array(pilots)
        super().__init__(mask, pilots, trainable=False, normalize=False,
                         dtype=pilot_pattern._dtype)
    
    @property
    def silent_ofdm_symbol_indices(self):
        return self._silent_ofdm_symbol_indices


# TODO
class MaskedPilotPattern(PilotPattern):
    """
    Taking another pilot pattern and a mask, a new pilot pattern is created.
    The mask masks (i.e. sets as pilot pattern with symbol 0) all RE where the mask is True.
    
    Parameters
    ----------
    pilot_pattern : PilotPattern
        Pilot pattern to be used as basis.
    mask : tf.Tensor
        Mask with shape [num_ofdm_symbols, num_effective_subcarriers].
    """
    
    def __init__(self, pilot_pattern, mask):
        self._internal_pilot_pattern = pilot_pattern
        self._mask = mask
        masking_mask = mask.numpy().astype(bool)
        pilots = pilot_pattern.pilots.numpy()

        # mask = np.expand_dims(mask, axis=(0,1))
        # mask = np.tile(mask, [pilots.shape[0], pilots.shape[1], 1, 1])
        # pilots[mask] = 0
        
        super().__init__(pilot_pattern.mask, pilots, trainable=False, normalize=False,
                         dtype=pilot_pattern._dtype)
  
Jammer Mitigation
===========================

Several jammer mitigation algorithms are implemented here.
Usage varies by algorithm, consult the documentation for each.
Every algorithm is integrated into the simulation code.

Projection onto Orthogonal Subspace (POS)
-----------------------------------------
.. automodule:: jammer.mitigation.POS
   :exclude-members: call
   :members:

LMMSE treating Interference as Noise (IAN)
------------------------------------------
.. automodule:: jammer.mitigation.IAN
   :members:

MitigAtion via Subspace Hiding (MASH)
-------------------------------------
.. automodule:: jammer.mitigation.MASH
   :exclude-members: call
   :members:

References:
   .. [Marti2023] `G. Marti and C. Studer, "Universal MIMO Jammer Mitigation via Secret Temporal Subspace Embeddings" <https://arxiv.org/abs/2305.01260>`_,
      arXiv e-prints. 2023
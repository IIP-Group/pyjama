#%%
import sionna
import numpy as np

p = sionna.utils.plotting.PlotBER()
p.add(np.arange(0, 10), 10.**-np.arange(1, 11))
p(ylim=(None, 1))
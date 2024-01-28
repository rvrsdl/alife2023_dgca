"""
The worked example described in paper 1 ("Developmental Graph Cellular Automata")
and shown in Figurs 1 and 2.
"""
#%%
import numpy as np
import matplotlib.pyplot as plt

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJ_DIR)
from src.dgca import DGCA, show_dgca

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5))


gm = np.array([[0,1,1,0,1],[0,0,0,1,0],[0,0,1,0,0],[1,0,0,0,0],[0,1,0,0,0]])
sm = np.array([[0,1,0],[0,0,1],[0,1,0],[1,0,0],[0,1,0]])
d = DGCA(gm,sm)
# set weights as in paper
d.weights = np.array([
[-0.2, -0.7, -0.3,  0.1, -0.8],
[-0.1, -0.1,  0.4,  0.4, -0.3],
[ 0.3, -0.8,  0.0,  0.5, -1.0],
[ 0.9, -0.4, -0.6,  0.9,  0.9],
[ 0.9, -0.6,  1.0, -0.4,  0.5],
[ 0.1,  0.7,  0.8,  0.7,  0.3],
[ 0.2,  0.3, -0.6,  0.1,  0.7]])
show_dgca(d, ax=axs[0])
axs[0].title.set_text('Step 0')

# %
# run one update step
d.update()
show_dgca(d, ax=axs[1])
axs[1].title.set_text('Step 1')

plt.show()


# %%

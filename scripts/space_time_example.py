
"""
Plotting a sample space-time diagram, as shown in
Figures 4 & 5 of paper 1 ("Developmental Graph Cellular Automata)
"""
#%%

import numpy as np
import matplotlib.pyplot as plt

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJ_DIR)
from src.dgca import *
from src.plotting import plot_space_time

start_size = 5
start_connectivity = 0.4
nstates = 3
gm = get_random_graph_mat(n=start_size, p=start_connectivity)
sm = get_random_state_mat(n=start_size, s=nstates)
d = DGCA(gm,sm)

max_size =256
max_steps = 20
hashes, steps = run_dgca(d,max_nodes=max_size, max_steps=max_steps)
plot_space_time(steps, show_components=True)
tl, al = get_trans_attr_len(hashes)
beh = get_behaviour_class(hashes, steps, max_size=max_size, max_steps=max_steps)
print(f'Transient Len: {tl}\nAttractor Len:{al}')
print(f'Behaviour Class: {beh}')
# %%

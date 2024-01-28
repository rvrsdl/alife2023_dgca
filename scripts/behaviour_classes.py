"""
Generating the data for Figure 3 of paper 1.
For systems with 2 states up to 9 states, we run 1000 trials
of development, and categorise the dynamical behaviour 
into five classes.
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJ_DIR)
from src.dgca import DGCA, get_ring_graph_mat, get_random_state_mat, Behaviour, run_dgca, get_behaviour_class
from src.plotting import plot_space_time

MAX_STEPS = 256
MAX_NODES = 256

gm = get_ring_graph_mat(n=8, self_loops=False)
results = dict()
for num_states in range(2,9):
    print(f'Running trials with {num_states} states')
    results[num_states] = {k:0 for k in Behaviour} # init dictionary
    for i in tqdm(range(1000)):
        sm = get_random_state_mat(n=8, s=num_states)
        d = DGCA(gm,sm)
        hashes, steps = run_dgca(d, max_steps=MAX_STEPS, max_nodes=MAX_NODES)
        behaviour = get_behaviour_class(hashes, steps, max_steps=MAX_STEPS, max_size=MAX_NODES)
        results[num_states][behaviour] += 1

out = [[r[b] for b in Behaviour] for r in results.values()]
plt.plot(out)
plt.legend([b.name for b in Behaviour])
plt.title('Occurences of Behaviour classes in 1000 trials')
plt.xlabel('Num states in system')
plt.show()

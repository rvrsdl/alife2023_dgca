"""
Comparing the behaviour of the systems (transient length, attractor cycle length and maximum size)

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
from src.dgca import DGCA, get_ring_graph_mat, get_sequential_state_mat, get_dgca_behaviour, run_dgca
from src.microbial_ga import MicrobialGA
from src.plotting import rand_jitter, plot_space_time
#%%
NSTATES = 8
MAX_STEPS = 256
MAX_NODES = 256
KNN = 5
MUT_RATE = 0.05
NTRIALS = 1000
POPSIZE = 30

# seed graph
gm = get_ring_graph_mat(n=8, self_loops=True)
sm = get_sequential_state_mat(n=8, s=NSTATES)

# random search
print('Running random search')
random_behaviour = []
for i in tqdm(range(NTRIALS)):
    dgca = DGCA(gm,sm)
    beh = get_dgca_behaviour(dgca, max_steps=MAX_STEPS, max_nodes=MAX_NODES)
    random_behaviour.append(beh)
random_behaviour = np.array(random_behaviour)

# novelty search
print('Running novelty search')
dgca = DGCA(gm,sm)
mga = MicrobialGA(pop_size=30, seed_dgca=dgca, max_steps=MAX_STEPS, max_nodes=MAX_NODES, knn=KNN, mut_rate=MUT_RATE)
mga.run(ntrials=NTRIALS-POPSIZE) # because the initial 30 population get added to the archive at the start.
novelty_behaviour = mga.get_archived_behaviour(only_attractors=False)

#%% plotting
rb = np.unique(random_behaviour[random_behaviour[:,1]>0,:], axis=0)
nb = np.unique(novelty_behaviour[novelty_behaviour[:,1]>0,:], axis=0)
# plot results
fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(12,6))
axs[0].set_box_aspect(1)
axs[1].set_box_aspect(1)
fig.subplots_adjust(wspace=0)
sc1 = axs[0].scatter(x=rand_jitter(rb[:,0]),
                    y=rand_jitter(rb[:,1]),
                    c=rb[:,2], marker='.')
axs[0].grid()
axs[0].set_xlabel('Transient Length')
axs[0].set_ylabel('Attractor Cycle Length')
axs[0].title.set_text(f'Random Search with {NSTATES} States')
sc2 = axs[1].scatter(x=rand_jitter(nb[:,0]),
                    y=rand_jitter(nb[:,1]),
                    c=nb[:,2], marker='.')
axs[1].grid()
axs[1].set_xlabel('Transient Length')
axs[1].set_ylabel('Attractor Cycle Length')
axs[1].title.set_text(f'Novelty Search with {NSTATES} States')
cbar = fig.colorbar(sc2, ax=axs)
cbar.ax.set_ylabel('Max Size')
# Plot max steps line on both
max_steps=256
if max_steps is not None:
    xl = axs[0].get_xlim()
    yl = axs[0].get_ylim()
    y0 = (max_steps-xl[0], max_steps-xl[1])
    y1 = (max_steps-xl[0], max_steps-xl[1])
    axs[0].plot(xl,y0,'r--')
    axs[1].plot(xl,y1,'r--')
    # rest ax limits
    axs[0].set_ylim(*yl)
    axs[1].set_ylim(*yl)
    axs[0].set_xlim(*xl)
    axs[1].set_xlim(*xl)
plt.show()
plt.savefig(f'random_vs_novelty_{NSTATES}state.svg')
# %%
# Plot space time diagram of some unusual behaviour
all_beh = mga.get_archived_behaviour(only_attractors=True)
all_dgca = mga.get_archived_dgcas(only_attractors=True)
longest_attr_idx = np.argmax(all_beh[:,1])
# br=1
# fit = mga.fitness_all()
# sortix = np.argsort(fit,)
# beh = all_beh[sortix[br]]
beh = all_beh[longest_attr_idx,:]
print(f'TransLen={beh[0]}\nAttrLen={beh[1]}\nMaxSize={beh[2]}')
hashes, steps = run_dgca(all_dgca[longest_attr_idx], max_steps=MAX_STEPS, max_nodes=MAX_NODES)
plot_space_time(steps, show_components=True, save=True)
# %%

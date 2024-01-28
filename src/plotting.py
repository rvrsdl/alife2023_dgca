"""
Contains plotting tools.
Space time diagram
Analysis of different behaviour classes.
"""
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJ_DIR)
from src.dgca import DGCA, get_trans_attr_len, get_behaviour_class

def plot_space_time(steps: list[DGCA], show_components: bool = False,
        save: bool = False):
    """
    Plots the uneven space-time diagram. If you supply the optional arguments it 
    puts the behaviour name in the title
    """
    state_vecs = [d.get_state_vec() for d in steps]
    if show_components:
        conn_comps = [d.get_connected_components() for d in steps]
        # now reorder so that connected component nodes are always together
        sortix = conn_comps_sortix(conn_comps)
        conn_comps = [elm[srt] for elm,srt in zip(conn_comps,sortix)]
        state_vecs = [elm[srt] for elm,srt in zip(state_vecs,sortix)]
        stacked = list(stack_uneven_multi(state_vecs, conn_comps).values())
        separators = np.cumsum([x.shape[1] for x in stacked])
        separators = separators[:-1]-0.5
        stacked = np.hstack(stacked)
    else:
        stacked = stack_uneven(state_vecs, fill=-99)# type: ignore
        separators = None
    stacked[stacked==-99] = np.nan
    plt.imshow(stacked)
    if separators is not None:
        ys=[-0.5,stacked.shape[0]-0.5]
        for x in separators:
            plt.plot([x,x], ys, 'r')
    if False:# behaviour:
        hashes = [d.get_hash() for d in steps]
        tl, al = get_trans_attr_len(hashes)
        get_behaviour_class(hashes, stepsss)
        title = 'Transient: %d; Attractor: %d; Behaviour: %s' % (
            behaviour.transient_length, behaviour.cycle_length, behaviour.get_behaviour_class())
        plt.title(title)
    plt.ylabel('Timesteps')
    plt.xlabel('Nodes')
    plt.show()
    # Save if necessary
    if save:
        dtime = datetime.datetime.now().strftime('%Y%m%dT%H%M')
        fname = f'space_time_{dtime}.svg'
        plt.savefig(fname, format='svg')

def conn_comps_sortix(conn_comps: list[np.ndarray]) -> list[np.ndarray]:
    """
    Returns sort indices for a list of connected component marker arrays.
    Since graphs can only break up and not rejoin, the last entry will
    be the most broken up. So we should begin by sorting that, then 
    go back up the list sorting earlier arrays, but biasing the sort 
    towards what we already have (by adding prev_sortix/1000 to the 
    flag arrays)
    """
    out = []
    sz = conn_comps[-1].shape[0]
    zz = np.array(range(sz))
    for cc in reversed(conn_comps):
        cc_plus = add_uneven_vec(cc, zz/1000, fill=0.99)
        sortix = np.argsort(cc_plus)
        out.append(sortix)
        zz = np.zeros(shape=cc.shape[0])
        zz[sortix] = np.array(range(cc.shape[0]))
    return list(reversed(out))

def add_uneven_vec(vec1: np.ndarray, vec2: np.ndarray, fill: float) -> np.ndarray:
    """
    Add vectors of unven length. The returned vector will be the
    length of vec1, with excess elements of vec2 dropped. If vec1
    is longer, then vec2 will be made up to the correct length by 
    filling with the fill value before addition.
    """
    len1 = vec1.shape[0]
    len2 = vec2.shape[0]
    if len1 == len2:
        # same size so no need to do anything special
        return np.add(vec1, vec2)
    elif len1 < len2:
        # drop extra entries of vec2
        return np.add(vec1, vec2[:len1])
    else:
        # fill vec2 ith fill value
        vec2_fill = np.concatenate((vec2,np.ones(shape=(len1-len2,))*fill))
        return np.add(vec1, vec2_fill)

def stack_uneven(list_of_vecs: list[np.ndarray], fill=np.nan):
    """
    Stacks an list of vectors of unequal length, padding the shorter ones 
    with nans at the right so they are the same size as the biggest.
    For uneven state time diagrams.
    """
    maxlen = max([len(v) for v in list_of_vecs])
    return np.vstack([np.concatenate([v, np.ones((maxlen-len(v),))*fill]) for v in list_of_vecs])

def stack_uneven_multi(
        vecs: list[np.ndarray],
        grps: list[np.ndarray],
        fill=np.nan) -> dict[int, np.ndarray]:
    grp_nums = np.unique(np.hstack(grps))
    separate = dict()
    for g in grp_nums:
        temp = []
        for vr,gr in zip(vecs, grps):
            temp.append(vr[gr==g])
        separate[g] = stack_uneven(temp,fill)
    return separate

def rand_jitter(arr):
    """
    Adds a small amount of jitter to each point,
    scaled to th range of the data.
    Copied from https://stackoverflow.com/questions/8671808/matplotlib-avoiding-overlapping-datapoints-in-a-scatter-dot-beeswarm-plot
    """
    scale = .01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * scale


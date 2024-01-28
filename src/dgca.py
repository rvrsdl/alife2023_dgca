"""
Copyright (c) 2023, Riversdale Waldegrave
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. 

Contains the main DGCA class, 
functions for running the system for a fixed number of steps,
functions for categorising the behaviour of the system.
"""
# %%
from __future__ import annotations
from enum import Enum
import numpy as np
from scipy.sparse.csgraph import connected_components
from multiset import FrozenMultiset as fms
from wrapt_timeout_decorator.wrapt_timeout_decorator import timeout
# for plotting:
import networkx as nx 
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes


class DGCA(object):

    def __init__(self, graph_mat: np.ndarray, state_mat: np.ndarray) -> None:

        self.graph_mat = graph_mat # N x N
        self.state_mat = state_mat # N x S
        self.nstates = state_mat.shape[1] # S
        # For debugging save a copy of the graph as it started out
        self.orig_graph_mat = np.copy(self.graph_mat)
        self.orig_state_mat = np.copy(self.state_mat)
        self.weights = np.random.uniform(low=-1.0, high=1.0,
                                         size=(2*self.nstates+1, self.nstates+2)) # (2S+1) x (S+2)
        self.delete_flag = self.nstates
        self.duplicate_flag = self.nstates+1

    def get_inp_mat_lapl(self) -> np.ndarray:
        # Graph Laplacian = diag(degree) - adjacency
        L_out = np.diag(np.sum(self.graph_mat, axis=1)) - self.graph_mat
        L_in = np.diag(np.sum(self.graph_mat.T, axis=1)) - self.graph_mat.T #should transpose graph_mat??
        F_out = L_out @ self.state_mat # N x S
        F_in = L_in @ self.state_mat # N x S
        bias = np.ones(shape=(self.graph_mat.shape[0], 1))
        return np.hstack([F_in, F_out, bias]) # N x (2S+1)
    
    def update(self, inplace: bool = True) -> DGCA | None:
        """
        This updates the DGCA and returns the t+1 version
        (NB does not modify in place)
        """
        inp_mat = self.get_inp_mat_lapl()
        out_mat = inp_mat @ self.weights # N x (S+2)
        action_choice = np.argmax(out_mat, axis=1)
        to_delete = action_choice == self.delete_flag
        to_keep = np.logical_not(to_delete)
        to_duplicate = action_choice == self.duplicate_flag
        # Duplicate the nodes we need to
        new_rows = np.copy(self.graph_mat[to_duplicate, :])[:, to_keep]
        new_cols = np.copy(self.graph_mat[:, to_duplicate])[to_keep, :]
        num_new = np.sum(to_duplicate)
        # the lower right part of the new matrix is all zeros 
        # because there can't be connections between new nodes
        lower_right = np.zeros(shape=(num_new, num_new))
        # the upper left part is the original graph - remove the deleted nodes
        upper_left = self.graph_mat[to_keep,:][:,to_keep]
        # Put it all together
        graph_mat = np.block([[upper_left, new_cols], [new_rows, lower_right]])
        # Deal with state matrix
        #   "original" nodes get original state, copied nodes get new state (arbitrary rule, but fine for now)
        new_state = np.max(out_mat[:,:-2], axis=1, keepdims=True) == out_mat[:,:-2]
        new_state = new_state.astype(np.float64)
        state_mat = np.vstack([new_state[to_keep,:], self.state_mat[to_duplicate,:]])
        if inplace:
            self.graph_mat = graph_mat
            self.state_mat = state_mat
        else:
            # Build new DGCA to return
            new_dgca = DGCA(graph_mat, state_mat)
            new_dgca.orig_graph_mat = self.orig_graph_mat
            new_dgca.orig_state_mat = self.orig_state_mat
            new_dgca.weights = self.weights
            return new_dgca
    
    def reset(self) -> None:
        """
        Resets graph_mat and state_mat to what they were originally.
        """
        self.graph_mat = np.copy(self.orig_graph_mat)
        self.state_mat = np.copy(self.state_mat)

    def get_hash(self) -> int:
        """
        Preliminary step in detecting attractors.
        Returns hash of node states plus theri neghbourhood data.
        """
        if self.graph_mat.shape[0]==0:
            # All nodes removed. Return -1 to indicate this.
            return -1
        else:
            info = np.hstack([self.graph_mat @ self.state_mat, 
                            self.graph_mat.T @ self.state_mat, 
                            np.argmax(self.state_mat, axis=1, keepdims=1)]) #type: ignore
            return hash(fms(map(tuple, info.tolist())))

    def get_size(self) -> int:
        """
        Simply returns the current number of nodes.
        """
        return self.graph_mat.shape[0]
    
    def get_state_vec(self) -> np.ndarray:
        """
        Converts the state_mat (one-hot encoded) into a 
        vector of state numbers. eg. 
        [[0,1,0],[1,0,0],[1,0,0],[0,0,1]]
        becomes [1,0,0,2]
        """
        return np.argmax(self.state_mat, axis=1)
        
    def get_connected_components(self):
        """
        Returns a number for each subgraph indicating which component
        it is part of.
        eg. [1,1,2,2,1] means nodes 0,1,4 form one connected component
        and nodes 2&3 form another.
        """
        # Treat graph as undirected because we don't care which 
        # way the edges are going for theses purposes.
        cc = connected_components(self.graph_mat, directed=False)
        return cc[1]
    
    def copy(self) -> DGCA:
        """
        Simply returns a deep copy of this DGCA.
        """
        out = DGCA(np.copy(self.graph_mat), np.copy(self.state_mat))
        out.weights = np.copy(self.weights)
        out.orig_graph_mat = np.copy(self.orig_graph_mat)
        out.orig_state_mat = np.copy(self.orig_state_mat)
        return out
    

def get_random_graph_mat(n: int, p: float) -> np.ndarray:
    """
    Returns a random Erdos-Renyi graph (with self-loops allowed),
    with n nodes and p probability of edges.
    """
    return np.random.choice([1,0], p=[p, 1-p], size=(n, n))

def get_ring_graph_mat(n: int, self_loops: bool) -> np.ndarray:
    ring = np.roll(np.eye(n), -1, axis=0)
    if self_loops:
        ring += np.eye(n)
    return ring

def get_random_state_mat(n: int, s: int) -> np.ndarray:
    """
    Returns a random state matrix of n rows (nodes) of
    one-hot state vectors with s entries.
    """
    rnd = np.random.uniform(size=(n,s))
    state_mat = rnd == np.max(rnd, axis=1, keepdims=True)
    return state_mat.astype(np.int32)

def get_sequential_state_mat(n: int, s: int) -> np.ndarray:
    """
    Returns a state matrix of n rows (nodes) of one-hot state
    vectors in sequential order. Repeats if n > s.
    """
    out = np.zeros(shape=(n,s))
    for i in range(n):
        out[i, i % s] = 1
    return out

def dgca_to_nx(dgca: DGCA) -> nx.DiGraph:
    g = nx.from_numpy_array(dgca.graph_mat, create_using=nx.DiGraph)
    state_vec = dgca.get_state_vec()
    st = {k:{'state':v} for k,v in zip(g.nodes, state_vec)}
    nx.set_node_attributes(g,st)
    return g

def show_dgca(dgca, ax: Axes | None = None) -> None:
    state_vec = dgca.get_state_vec()
    nnodes, nstates = dgca.state_mat.shape
    g = nx.from_numpy_array(dgca.graph_mat, create_using=nx.DiGraph)
    lab = {z[0]:'%d:%s' % z for z in zip(range(nnodes), state_vec)}
    state_colours = plt.get_cmap('viridis',lut=nstates).colors #type: ignore
    node_colours = [state_colours[n] for n in state_vec]
    pos = nx.spring_layout(g)
    nx.draw_networkx_nodes(g, pos, node_color=node_colours, node_size=100, ax=ax)  # type: ignore
    nx.draw_networkx_labels(g, pos, labels=lab, font_size=8, ax=ax)
    nx.draw_networkx_edges(
        g, pos,
        connectionstyle="arc3,rad=0.1",
        node_size=100,
        ax = ax
    )

@timeout(5, use_signals=False)
def check_iso_states(dgca1: DGCA, dgca2: DGCA) -> bool:
    """
    Checks if two graphs are isomorphic AND have the same node states
    """
    g1 = dgca_to_nx(dgca1)
    g2 = dgca_to_nx(dgca2)
    node_match_func = nx.algorithms.isomorphism.categorical_node_match('state',-99)
    is_iso = nx.is_isomorphic(g1, g2, node_match=node_match_func)
    return is_iso

def run_dgca(dgca: DGCA,
             max_steps: int = 20,
             max_nodes: int = 256
             ) -> tuple[list[int], list[DGCA]]:
    """
    Runs a DCGA until either max_steps or max_nodes is reached.
    Stops early if an attractor is found (ie. if we get a graph+states
    that we have seen before).
    Returns a list of hashes of the graphs (good for spotting attractor
    cycles etc.), and the graphs (DGCA objects) themselves.
    """
    all_steps = []
    hashes = []
    h = dgca.get_hash()
    keepgoing = True
    for i in range(max_steps):
        if (h in hashes):
            possible_match = all_steps[hashes.index(h)]
            try:
                is_isomorphic = check_iso_states(dgca, possible_match)
            except TimeoutError:
                print('Isomorphism check timed out - ending early')
                is_isomorphic = True
                h = -2 # my convention to indicate this has happened.
            if is_isomorphic:
                keepgoing = False
            else:
                # It is in fact not isomorphic, so store under a different hash (by hashing the string of the current hash!)
                h = hash(str(h))
        if dgca.get_size() > max_nodes:
            keepgoing = False
        # append it to the record
        all_steps.append(dgca)
        hashes.append(h)
        if keepgoing:
            dgca = dgca.update(inplace=False) # type: ignore
            h = dgca.get_hash()
        else:
            break
    return hashes, all_steps

def get_trans_attr_len(hashes: list[int]) -> tuple[int,int]:
    """
    Takes the output of run_dgca (hashes and list)
    Returns a tuple of (transient_length, cycle_length, max_size)
    If no attractor was found, 0 is returned for the attractor cycle length.
    """
    if hashes[-1] in hashes[:-1]:
        trans_len = hashes[:-1].index(hashes[-1])
        attr_len = len(hashes) - trans_len - 1
    else:
        trans_len = len(hashes)
        attr_len = 0
    return trans_len, attr_len

Behaviour = Enum('Behaviour',['Dies','Runaway','Static','Halts','Slow','Unknown'])
def get_behaviour_class(hashes: list[int], steps: list[DGCA], max_size: int, max_steps: int):
    """
    Returns one of the five behaviour classes.
    """
    sizes = [d.get_size() for d in steps]
    if hashes[-1]==-1:
        # A hash value of -1 means all nodes were removed
        return Behaviour.Dies
    elif hashes[-1]==-2:
        # A hash value of -2 means the isomorphism check timed out so we ended early
        return Behaviour.Unknown
    elif max(sizes) > max_size:
        return Behaviour.Runaway
    elif np.std(sizes)==0:
        # size didn't change at all
        return Behaviour.Static
    elif hashes[-1] in hashes[:-1]:
        # attractor found
        return Behaviour.Halts
    elif len(hashes) >= max_steps:
        # No attractor found in time
        return Behaviour.Slow

def get_dgca_behaviour(dgca, max_steps: int, max_nodes: int) -> tuple[int,int,int]:
    """
    Helper function which runs a DGCA and analyses its dynamical
    behaviour in one go.
    Returns tuple of (transient_length, attractor_cycle_length, max_size).
    These are the three dimensions of the behaviour space used by the
    novelty search.
    """
    hashes, steps = run_dgca(dgca, max_steps=max_steps, max_nodes=max_nodes)
    trans_len, attr_len = get_trans_attr_len(hashes)
    max_sz = max([d.get_size() for d in steps])
    return trans_len, attr_len, max_sz



"""
Simple implementation of the Microbial GA algorithm 
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJ_DIR)
from src.dgca import DGCA, get_dgca_behaviour, get_ring_graph_mat, get_random_state_mat, Behaviour, run_dgca, get_behaviour_class, get_trans_attr_len


def crossover_cols(mat1: np.ndarray, mat2: np.ndarray) -> np.ndarray:
    """
    Crosses over two weights matrices by column.
    """
    mat_out = np.zeros(shape=mat1.shape)
    for c in range(mat_out.shape[1]):
        # 50/50 chance of column being used from each matrix
        if np.random.uniform()<0.5:
            mat_out[:,c] = mat1[:,c]
        else:
            mat_out[:,c] = mat2[:,c]
    return mat_out

def crossover_rows(mat1: np.ndarray, mat2: np.ndarray) -> np.ndarray:
    """
    Crosses over two weights matrices by column.
    """
    mat_out = np.zeros(shape=mat1.shape)
    for c in range(mat_out.shape[0]):
        # 50/50 chance of column being used from each matrix
        if np.random.uniform()<0.5:
            mat_out[c,:] = mat1[c,:]
        else:
            mat_out[c,:] = mat2[c,:]
    return mat_out

def mutate(mat_in: np.ndarray, mut_rate: float) -> np.ndarray:
    """
    Changes some randomly selected values in the matrix to 
    new values (from -1,1 uniform distribution).
    """
    mask = np.random.choice([True, False], p=[mut_rate, 1-mut_rate], size=mat_in.shape)
    mat_out = np.copy(mat_in)
    new_vals = np.random.uniform(low=-1.0, high=1.0, size=mat_in.shape)
    mat_out[mask] = new_vals[mask]
    return mat_out

class MicrobialGA():

    def __init__(self, 
                 pop_size: int,
                 seed_dgca: DGCA,
                 max_steps: int,
                 max_nodes: int,
                 knn: int,
                 mut_rate: float
                 ) -> None:
        self.pop_size = pop_size
        self.seed_dgca = seed_dgca
        self.population = [DGCA(np.copy(seed_dgca.graph_mat), np.copy(seed_dgca.state_mat)) for _ in range(pop_size)]
        self.max_steps = max_steps
        self.max_nodes = max_nodes
        # Initialise archive (dict: key = initial dgca; value = behaviour)
        self.archive = {d: get_dgca_behaviour(d, max_steps, max_nodes) for d in self.population}
        self.knn = knn
        self.mut_rate = mut_rate

    def contest(self) -> None:
        idx1, idx2 = np.random.choice(range(self.pop_size), size=2, replace=False)
        if self.population[idx1] in self.archive:
            behaviour1 = self.archive[self.population[idx1]]
        else:
            # This shouldn't happen as we are adding them to the archive on creation.
            raise ValueError('DGCA not found in archive')
        if self.population[idx2] in self.archive:
            behaviour2 = self.archive[self.population[idx2]]
        else:
            # This shouldn't happen as we are adding them to the archive on creation.
            raise ValueError('DGCA not found in archive')
        fitness1 = self.fitness_single(behaviour1)
        fitness2 = self.fitness_single(behaviour2)
        if fitness1 >= fitness2: #type: ignore
            win_idx, lose_idx = idx1, idx2
        else:
            win_idx, lose_idx = idx2, idx1
        offspring = DGCA(np.copy(self.seed_dgca.graph_mat), np.copy(self.seed_dgca.state_mat))
        offspring.weights = mutate(
            crossover_rows(
                self.population[win_idx].weights,
                self.population[lose_idx].weights),
            mut_rate=self.mut_rate)
        # replace loser with offspring
        self.population[lose_idx] = offspring
        # find offspring behaviour straight away
        offspring_behaviour = get_dgca_behaviour(offspring, max_steps=self.max_steps, max_nodes=self.max_nodes)
        # add to archive
        self.archive[offspring] = offspring_behaviour
        # threshold for adding? reduce archive?

    def get_archived_behaviour(self, only_attractors: bool) -> np.ndarray:
        """
        Returns all the behaviours in the archive as a numpy array.
        One row per behaviour, cols are dimensions.
        """
        beh = np.array([b for b in self.archive.values()])
        if only_attractors:
            # Only return behaviour where an attractor was found (attractro)
            beh = beh[beh[:,1]>0,:]
        return beh
    
    def get_archived_dgcas(self, only_attractors: bool) -> np.ndarray:
        beh = np.array([b for b in self.archive.values()])
        dgcas = np.array([d for d in self.archive.keys()])
        if only_attractors:
            dgcas = dgcas[beh[:,1]>0]
        return dgcas

    def fitness_single(self, behaviour: tuple[int,int,int]) -> np.floating:
        """
        Returns novelty (fitness) score of a single behaviour
        based on mean distance to KNN behaviours in the archive.
        """
        if behaviour[1]==0:
            # zero fitness if no attractor found
            return 0 #type: ignore
        else:
            compareto = self.get_archived_behaviour(only_attractors=True)
            nbrs = NearestNeighbors(n_neighbors=self.knn, algorithm='ball_tree').fit(compareto)
            dists, _ = nbrs.kneighbors(np.array([behaviour]))
            return np.mean(dists)
    
    def fitness_all(self) -> np.ndarray:
        """
        Find the fitness of all the behaviours in the archive.
        """
        compareto = self.get_archived_behaviour(only_attractors=True)
        all_behaviour = self.get_archived_behaviour(only_attractors=False)
        nbrs = NearestNeighbors(n_neighbors=self.knn, algorithm='ball_tree').fit(compareto)
        dists, _ = nbrs.kneighbors(all_behaviour)
        fitness = np.mean(dists, axis=1)
        # zero fitness if no attractor found
        fitness[all_behaviour[:,1]==0] = 0
        return fitness

    def run(self, ntrials: int):
        """
        Runs the Microbial GA for n trials
        """
        for i in tqdm(range(ntrials)):
            self.contest()

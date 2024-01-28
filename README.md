# Developmental Graph Cellular Automata (ALife 2023)
This repository contains the code used in the following two papers:

- [1] R. Waldegrave, S. Stepney, and M. A. Trefzer, ‘Developmental Graph Cellular Automata’, presented at the ALIFE 2023: Ghost in the Machine: Proceedings of the 2023 Artificial Life Conference, MIT Press, Jul. 2023. doi: 10.1162/isal_a_00658. [Link](https://direct.mit.edu/isal/proceedings/isal/35/55/116871)
- [2] R. Waldegrave, S. Stepney, and M. A. Trefzer, ‘Exploring the Rich Behaviour of Developmental Graph Cellular Automata’, presented at the ALIFE 2023: Ghost in the Machine: Proceedings of the 2023 Artificial Life Conference, MIT Press, Jul. 2023. doi: 10.1162/isal_a_00666. [Link](https://direct.mit.edu/isal/proceedings/isal/35/61/116896)

The repository is organised as follows:
- `src/` contains:
    -`dgca.py`: code for the DGCA system
    - `microbial_ga.py`: a very simple implementation of a Microbial Genetic Algorithm for use in Novelty Search
    - `plotting.py`: some functions for plotting space-time diagrams etc.
- `scripts/` contains:
    - `worked_example.py`: code for the worked example in the first paper [1], going through how a single update step of the system works. **This is a good place to start.**
    - `space_time_example.py`: running a DGCA system for a fixed number of timesteps and plotting the behaviour as a space-time diagram.
    - `behaviour classes`: measuring the occurence of five different classes of dynamical behaviour of the system,as set out in the first paper [1].
    - `random_vs_novelty_search.py`: comparing the behaviour of the system found with random search vs novelty search.
    
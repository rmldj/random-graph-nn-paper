import networkx as nx
import numpy as np


def straight_line(N, seed=0):
    """
    Creats a simple chain-like net.
    :param N: number of nodes.
    :param seed: the sedd.
    :return:
    """
    np.random.seed(seed)
    G = nx.DiGraph()
    G.add_nodes_from(range(N))

    for i in range(N - 1):
        G.add_edge(i, i + 1)

    return G

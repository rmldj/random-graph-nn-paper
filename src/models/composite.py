import networkx as nx
import numpy as np

from src.models import graph_utils


def pathhist(G):
    '''
    npaths[i] = number of paths of length i+1 between first and last node
    can be orders of magnitude faster than
    paths_lengths = np.array([len(p)-1 for p in nx.all_simple_paths(G, 0, n-1)])
    np.unique(paths_lengths, return_counts=True)
    '''
    n = G.number_of_nodes()
    adj = nx.to_numpy_array(G, nodelist=range(n))
    A = np.array(adj)
    npaths = np.zeros(n, dtype=int)
    for i in range(n):
        npaths[i] = A[0, n - 1]
        A[:, :] = np.dot(adj, A)
    return npaths


def logpaths(G):
    return np.log(np.sum(pathhist(G)))


def func(G):
    """
    The function being optimized by composite graphs.
    """
    n = G.number_of_nodes()
    return (logpaths(G) / n) ** 0.5 - 2 * nx.global_reaching_centrality(G) - nx.average_clustering(G)


def composite(N, p=0.99, seed=0):
    """
    Creates the composite graph.
    :param N: number of nodes.
    :param p: probability of connection for the initial ER graph.
    :param seed: the seed.
    :return: networkx graph, nodes positions.
    """
    current_seed = seed
    stuck = True  # for some seeds optimization does not work, then we change the seed until it works
    while stuck:
        G = nx.erdos_renyi_graph(N, p, seed=current_seed)
        posG = nx.drawing.kamada_kawai_layout(G)
        H, posH, rH = graph_utils.make_dag(G, posG)

        np.random.seed(current_seed)

        fh = func(H)

        for i in range(5000):
            edges = list(H.edges)
            e = edges[np.random.choice(len(edges))]
            H1 = H.copy()
            H1.remove_edges_from([e])
            graph_utils.fix_orphans(H1)
            fh1 = func(H1)
            if fh1 > fh:
                stuck = False
                H = H1
                fh = fh1

        if stuck:
            current_seed += 1

    print('composite {} {} {} {}'.format(N, p, current_seed, fh))
    return H, posH

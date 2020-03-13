import bisect

import networkx as nx
import numpy as np


def one(x):
    return 1.0


def exp2(x):
    return np.exp(-2 * x)


def exp3(x):
    return np.exp(-3 * x)


def pow1(x):
    return 1 / (1.0 + x)


def random_dag(N, nout_dist='constant', nout_par=3, B=5, alpha=0.5, func='exp2', seed=0, verbose=False):
    """
    Creates a random DAG.
    :param N: number of nodes.
    :param nout_dist: the distribution on the outgoing edges.
    :param nout_par: number of
    :param B:
    :param alpha:
    :param func:
    :param seed:
    :param verbose:
    :return:
    """
    np.random.seed(seed)

    G = nx.DiGraph()
    G.add_nodes_from(range(N))

    if nout_dist == 'constant':
        nout = nout_par * np.ones(N, dtype=int)
    if nout_dist == 'laplace':
        nout = 1 + np.fabs(np.random.laplace(scale=nout_par, size=N)).astype(int)
    if nout_dist == 'fixed':
        nout = nout_par  # has to be an array of ints
    if nout_dist == 'hubs':
        c, positions, hubc = nout_par
        nout = c * np.ones(N, dtype=int)
        for p in positions:
            nout[p] = hubc

    for i in range(N - 1):
        nout[i] = min(nout[i], N - i - 1)

    f = globals()[func]

    for i in range(N - 1):
        if verbose:
            print(i, nout[i])
        nedges = nout[i]
        S = list(range(i + 1, N))
        if len(list(G.predecessors(i + 1))) == 0:
            G.add_edge(i, i + 1)
            S.remove(i + 1)
            nedges = nedges - 1
        if nedges == 0:
            continue
        for k in range(nedges):
            weight = nout[S] ** alpha * f(np.trunc((np.array(S) - i) / B))
            w = weight / np.sum(weight)
            thresholds = np.cumsum(w)
            r = np.random.uniform()
            j = bisect.bisect(thresholds, r)
            G.add_edge(i, S[j])
            S.remove(S[j])

    return G

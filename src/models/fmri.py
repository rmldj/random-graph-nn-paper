import random

import Graph_Sampling
import networkx as nx
import nibabel as nii
import numpy as np
import os
from scipy.sparse.csgraph import connected_components

# MODIFY THIS PATH
HCP_50_PATH = os.path.abspath("./data/netmats/3T_HCP1200_MSMAll_d50_ts2/Mnet2.pconn.nii")
HCP_100_PATH = os.path.abspath("./data/netmats/3T_HCP1200_MSMAll_d100_ts2/Mnet2.pconn.nii")

net50 = nii.load(HCP_50_PATH)
arr50 = net50.get_fdata()

net100 = nii.load(HCP_100_PATH)
arr100 = net100.get_fdata()


def get_adj(arr, t):
    """
    get the adjacency matrix (by thresholding)
    :param arr: the correlation matrix.
    :param t: the threshold.
    :return: the adjacency matrix.
    """
    adj = np.zeros(arr.shape, dtype=int)
    adj[np.fabs(arr) > t] = 1
    return adj


def get_graph(arr, t):
    """
    Get the fmri graph by selecting the largest connected component.
    :param arr: the correlation matrix.
    :param t: the threshold.
    :return: the adjacency matrix for the selected subgraph.
    """
    adj = get_adj(arr, t)
    n_components, labels = connected_components(adj, directed=False, return_labels=True)
    ids, counts = np.unique(labels, return_counts=True)
    c = np.argmax(counts)
    label = ids[c]
    mask = (labels == label)
    return adj[mask, :][:, mask]


def fmri(n, threshold, seed):
    """
    Creates the fmri-based graph.
    :param n: number of nodes.
    :param threshold: the threshold (see paper for definition).
    :param seed: the seed.
    :return: networkx graph based on the fMRI-data.
    """
    assert n == 30 or n == 60
    if n == 30:
        arr = arr50
    if n == 60:
        arr = arr100
    G = nx.Graph(get_graph(arr, threshold))
    assert G.number_of_nodes() >= n

    random.seed(seed)

    #perform subsampling.

    obj = Graph_Sampling.SRW_RWF_ISRW()
    G_subsampled = obj.random_walk_induced_graph_sampling(G, n)

    return nx.convert_node_labels_to_integers(G_subsampled)

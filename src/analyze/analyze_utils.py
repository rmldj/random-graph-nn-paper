import importlib
import sys

import networkx as nx
import numpy as np
from scipy.linalg import expm

import src.analyze.network_features as nf
from src.models import graph_utils as gu

"""This file is used to compute the selected features. For the definition please refer to the features documentation. 
"""

def graphinfo(s, net_dir='./graphs'):
    sys.path.append(net_dir)
    Net = getattr(importlib.import_module(s), 'Net')
    G = nx.DiGraph()
    G.add_nodes_from(range(Net.num_nodes))
    G.add_edges_from(Net.edges)
    info = dict()
    info['G'] = G
    info['meta'] = Net.meta
    info['stages'] = Net.stages
    info['num_nodes'] = Net.num_nodes
    info['num_units'] = Net.num_units
    info['edges'] = Net.edges
    info['pos'] = Net.pos
    return info


stagedicts = dict()

columnsbis = ['num_nodes', 'num_edges', 'reduce_frac', 'edges_per_node', 'density', 'transitivity',
              'average_clustering', 'average_node_connectivity', 'average_shortest_path_length',
              's_metric_norm', 'global_reaching_centrality', 'edge_connectivity', 'modularity_trace',
              'intrastage', 'interstage', 'hops_per_node', 'mean_degree', 'std_degree', 'span_degree',
              '021D', '021U', '021C', '030T',
              'log_paths', 'mean_path', 'std_paths', 'min_path', 'max_path', 'span_path']


def get_stage_dict(n, stages):
    pair = (n, tuple(stages))
    if pair in stagedicts:
        return stagedicts[pair]
    else:
        stagedict = gu.get_stage_dict(n, stages)
        stagedicts[pair] = stagedict
        return stagedict


def pathhist(G):
    '''
    npaths[i] = number of paths of length i+1 between first and last node
    can be orders of magnitude faster than
    paths_lengths = np.array([len(p)-1 for p in nx.all_simple_paths(G, 0, n-1)])
    np.unique(paths_lengths, return_counts=True)
    '''
    n = G.number_of_nodes()
    adj = nx.to_numpy_array(G)
    A = np.array(adj)
    npaths = np.zeros(n, dtype=int)
    for i in range(n):
        npaths[i] = A[0, n - 1]
        A[:, :] = np.dot(adj, A)

    return npaths


def features_part0(info):
    """
    first set of features.
    """
    G = info['G']
    posG = info['pos']
    stages = info['meta']['stages']

    res = dict()
    res['max_path'] = nf.max_path(G, posG, stages)
    res['outter_edges'] = nf.outter_edges(G, posG, stages)
    res['mean_degree'] = nf.mean_degree(G, posG, stages)
    res['mean_in_degree'] = nf.mean_in_degree(G, posG, stages)
    res['mean_out_degree'] = nf.mean_out_degree(G, posG, stages)
    res['max_degree'] = nf.max_degree(G, posG, stages)
    res['min_degree'] = nf.min_degree(G, posG, stages)
    res['average_clustering'] = nf.average_clustering(G, posG, stages)
    res['degree_assortativity'] = nf.degree_assortativity(G, posG, stages)
    return res


def features_part1(info):
    """
    second set of features.
    """
    G = info['G']
    n = info['num_nodes']
    num_units = info['num_units']
    edges = info['edges']
    nedges = len(edges)

    res = dict()
    res['num_nodes'] = n
    res['num_edges'] = nedges
    res['reduce_frac'] = num_units / n - 1
    res['edges_per_node'] = nedges / n
    res['density'] = nx.density(G)
    res['transitivity'] = nx.transitivity(G)
    res['average_clustering'] = nx.average_clustering(G)
    res['average_node_connectivity'] = nx.average_node_connectivity(G)
    res['average_shortest_path_length'] = nx.average_shortest_path_length(G)
    res['s_metric_norm'] = np.sqrt(nx.s_metric(G, normalized=False) / nedges)
    res['global_reaching_centrality'] = nx.global_reaching_centrality(G)
    res['edge_connectivity'] = nx.edge_connectivity(G, 0, n - 1)
    res['modularity_trace'] = np.real(np.sum(nx.modularity_spectrum(G)))

    stages = info['stages']
    stagedict = get_stage_dict(n, stages)
    edges_diff = np.array([stagedict[j] - stagedict[i] for (i, j) in edges])
    n0 = np.sum(edges_diff == 0)
    n1 = np.sum(edges_diff == 1)
    n2 = np.sum(edges_diff == 2)

    res['intrastage'] = n0 / nedges
    res['interstage'] = n1 / nedges
    res['hops_per_node'] = n2 / n

    degrees = np.array(nx.degree(G))[:, 1]
    res['mean_degree'] = np.mean(degrees)
    res['std_degree'] = np.std(degrees)
    res['span_degree'] = np.amax(degrees) / np.amin(degrees)

    triadic = nx.triadic_census(G)
    res['021D'] = triadic['021D'] / nedges
    res['021U'] = triadic['021U'] / nedges
    res['021C'] = triadic['021C'] / nedges
    res['030T'] = triadic['030T'] / nedges

    paths_nums = pathhist(G)
    ns = np.arange(1, n + 1)
    paths_total = np.sum(paths_nums)
    mean_path = np.sum(ns * paths_nums) / paths_total
    mean_pathsqr = np.sum(ns ** 2 * paths_nums) / paths_total
    std_path = np.sqrt(mean_pathsqr - mean_path ** 2)

    nonz = np.nonzero(paths_nums)[0]
    shortest_path = nonz[0] + 1
    longest_path = nonz[-1] + 1

    res['log_paths'] = np.log(paths_total)
    res['mean_path'] = mean_path
    res['std_paths'] = std_path
    res['min_path'] = shortest_path
    res['max_path'] = longest_path
    res['span_path'] = longest_path / shortest_path

    return res


def features_part2(info):
    """
    third set of features.
    """
    G = info['G']
    n = info['num_nodes']
    num_units = info['num_units']
    edges = info['edges']
    nedges = len(edges)

    H = G.to_undirected()

    res = dict()
    cc = nx.closeness_centrality(G)
    res['closeness_centrality'] = cc[n - 1]
    res['closeness_centrality_mean'] = np.mean(list(cc.values()))

    bc = nx.betweenness_centrality(G)
    res['betweenness_centrality_mean'] = np.mean(list(bc.values()))

    cfcc = nx.current_flow_closeness_centrality(H)
    res['current_flow_closeness_centrality_mean'] = np.mean(list(cfcc.values()))

    cfbc = nx.current_flow_betweenness_centrality(H)
    res['current_flow_betweenness_centrality_mean'] = np.mean(list(cfbc.values()))

    soc = nx.second_order_centrality(H)
    res['second_order_centrality_mean'] = np.mean(list(soc.values())) / n

    cbc = nx.communicability_betweenness_centrality(H)
    res['communicability_betweenness_centrality_mean'] = np.mean(list(cbc.values()))

    comm = nx.communicability(H)
    res['communicability'] = np.log(comm[0][n - 1])
    res['communicability_start_mean'] = np.log(np.mean(list(comm[0].values())))
    res['communicability_end_mean'] = np.log(np.mean(list(comm[n - 1].values())))

    res['radius'] = nx.radius(H)
    res['diameter'] = nx.diameter(H)
    res['local_efficiency'] = nx.local_efficiency(H)
    res['global_efficiency'] = nx.global_efficiency(H)
    res['efficiency'] = nx.efficiency(H, 0, n - 1)

    pgr = nx.pagerank_numpy(G)
    res['page_rank'] = pgr[n - 1]
    res['page_rank_mean'] = np.mean(list(pgr.values()))

    cnstr = nx.constraint(G)
    res['constraint_mean'] = np.mean(list(cnstr.values())[:-1])

    effsize = nx.effective_size(G)
    res['effective_size_mean'] = np.mean(list(effsize.values())[:-1])

    cv = np.array(list(nx.closeness_vitality(H).values()))
    cv[cv < 0] = 0
    res['closeness_vitality_mean'] = np.mean(cv) / n

    res['wiener_index'] = nx.wiener_index(H) / (n * (n - 1) / 2)

    A = nx.to_numpy_array(G)
    expA = expm(A)
    res['expA'] = np.log(expA[0, n - 1])
    res['expA_mean'] = np.log(np.mean(expA[np.triu_indices(n)]))

    return res

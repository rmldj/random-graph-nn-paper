import networkx as nx
import numpy as np

from src.models import graph_utils as gu


def max_path(G, posG, stages, stage=None):
    """
    Computes the maximum path length.
    """
    Graph = G if stage is None else gu.get_subgraph(G, posG, stages, stage)
    return nx.algorithms.dag.dag_longest_path_length(Graph)


def average_shortest_path(G, posG, stages, stage=None):
    """
    Computes the average shortest path length.
    """
    Graph = G if stage is None else gu.get_subgraph(G, posG, stages, stage)
    return nx.algorithms.shortest_paths.generic.average_shortest_path_length(Graph)


def inner_edges(G, posG, stages, stage=None):
    """
    Computes the relative number of edges within stages.
    """
    if stage is not None:
        Graph = gu.get_subgraph(G, posG, stages, stage)
        return len(Graph.edges) / len(G.edges)
    else:
        num_nodes = G.number_of_nodes()
        stage_dict = gu.get_stage_dict(num_nodes, stages)
        count = 0
        for edge in G.edges():
            if stage_dict[edge[0]] == stage_dict[edge[1]]:
                count += 1
        edges_diff = np.array([stage_dict[j] - stage_dict[i] for (i, j) in G.edges()])
        return count / len(G.edges)


def outter_edges(G, posG, stages, stage=None):
    """
    Computes the relative number of edges crossing different stages.
    """
    num_nodes = G.number_of_nodes()
    stage_dict = gu.get_stage_dict(num_nodes, stages)
    if stage is not None:
        Graph = gu.get_subgraph(G, posG, stages, stage)
    else:
        Graph = G

    count = 0
    for edge in G.edges():
        if edge[0] in Graph.nodes() or edge[1] in Graph.nodes():
            if stage_dict[edge[0]] != stage_dict[edge[1]]:
                count += 1
    return count / len(G.edges)


def in_edges(G, posG, stages, stage=None):
    """
    Computes the number of input edges (useful when stage is not None)
    """
    if stage is None:
        return 0
    num_nodes = G.number_of_nodes()
    stage_dict = gu.get_stage_dict(num_nodes, stages)
    Graph = gu.get_subgraph(G, posG, stages, stage)

    count = 0
    for edge in G.edges():
        if edge[0] not in Graph.nodes() and edge[1] in Graph.nodes():
            count += 1
    return count / len(G.edges)


def out_edges(G, posG, stages, stage=None):
    """
    Computes the number of output edges (useful when stage is not None)
    """
    if stage is None:
        return 0
    num_nodes = G.number_of_nodes()
    stage_dict = gu.get_stage_dict(num_nodes, stages)
    Graph = gu.get_subgraph(G, posG, stages, stage)

    count = 0
    for edge in G.edges():
        if edge[0] in Graph.nodes() and edge[1] not in Graph.nodes():
            count += 1
    return count / len(G.edges)


def mean_degree(G, posG, stages, stage=None):
    """
    Computes the mean degree.
    """
    num_nodes = G.number_of_nodes()
    stage_dict = gu.get_stage_dict(num_nodes, stages)
    if stage is None:
        return np.mean(np.array(list(G.degree())), axis=0)[1]
    else:
        Graph = gu.get_subgraph(G, posG, stages, stage)
    degree_view = G.degree()
    res = 0.0
    for view in degree_view:
        if view[0] in Graph.nodes():
            res += view[1]
    return res / len(Graph.nodes())


def mean_in_degree(G, posG, stages, stage=None):
    """
    Computes the mean indegree.
    """
    num_nodes = G.number_of_nodes()
    stage_dict = gu.get_stage_dict(num_nodes, stages)
    if stage is None:
        return np.mean(np.array(list(G.in_degree())), axis=0)[1]
    else:
        Graph = gu.get_subgraph(G, posG, stages, stage)
    degree_view = G.in_degree()
    res = 0.0
    for view in degree_view:
        if view[0] in Graph.nodes():
            res += view[1]
    return res / len(Graph.nodes())


def mean_out_degree(G, posG, stages, stage=None):
    """
    Computes the mean outdegree.
    """
    num_nodes = G.number_of_nodes()
    stage_dict = gu.get_stage_dict(num_nodes, stages)
    if stage is None:
        return np.mean(np.array(list(G.out_degree())), axis=0)[1]
    else:
        Graph = gu.get_subgraph(G, posG, stages, stage)
    degree_view = G.out_degree()
    res = 0.0
    for view in degree_view:
        if view[0] in Graph.nodes():
            res += view[1]
    return res / len(Graph.nodes())


def max_degree(G, posG, stages, stage=None):
    """
    Computes the maximum degree.
    """
    num_nodes = G.number_of_nodes()
    stage_dict = gu.get_stage_dict(num_nodes, stages)
    if stage is None:
        return np.max(np.array(list(G.degree())), axis=0)[1]
    else:
        Graph = gu.get_subgraph(G, posG, stages, stage)
    degree_view = G.degree()
    res = -1 * np.inf
    for view in degree_view:
        if view[0] in Graph.nodes() and res < view[1]:
            res = view[1]
    return res


def min_degree(G, posG, stages, stage=None):
    """
    Computes the minimum degree.
    """
    num_nodes = G.number_of_nodes()
    stage_dict = gu.get_stage_dict(num_nodes, stages)
    if stage is None:
        return np.min(np.array(list(G.degree())), axis=0)[1]
    else:
        Graph = gu.get_subgraph(G, posG, stages, stage)
    degree_view = G.degree()
    res = np.inf
    for view in degree_view:
        if view[0] in Graph.nodes() and res > view[1]:
            res = view[1]
    return res


def average_clustering(G, posG, stages, stage=None):
    """
    Computes the average clustering coefficient.
    """
    Graph = G if stage is None else gu.get_subgraph(G, posG, stages, stage)
    return nx.average_clustering(Graph)


def degree_assortativity(G, posG, stages, stage=None):
    """
    Computes the degree assortativity (given by pearson correlation).
    """
    Graph = G if stage is None else gu.get_subgraph(G, posG, stages, stage)
    return nx.degree_pearson_correlation_coefficient(Graph)


# def average_node_connectivity(G, posG, stages, stage=None):
#    Graph = G if stage is None else gu.get_subgraph(G,posG,stages,stage)
#    return nx.average_node_connectivity(Graph)

def s_metric(G, posG, stages, stage=None):
    """
    Computes the s-metric (not normalized).
    """
    Graph = G if stage is None else gu.get_subgraph(G, posG, stages, stage)
    return nx.s_metric(Graph, normalized=False)


def number_of_paths_to_each_node(G):
    """
    Computes the number of paths to each node.
    :param G: the graph
    :return: array of paths (paths[i] = number of paths to node i), number of zero_indegree nodes.
    """
    sort = nx.topological_sort(G)
    paths = {n: 0 for n in G.nodes()}
    in_degree = G.in_degree()
    zero_degree = 0
    for i, n in enumerate(sort):
        if in_degree[n] == 0:
            paths[n] = 1
            zero_degree += 1
        for neighbour in G[n]:
            paths[neighbour] += paths[n]
    return paths, zero_degree


def number_of_paths(G, posG, stages, stage=None):
    """
    Computes the number of all paths in the graph.
    """
    Graph = G if stage is None else gu.get_subgraph(G, posG, stages, stage)
    paths, zero_degree = number_of_paths_to_each_node(Graph)
    return np.sum(list(paths.values())) - zero_degree


def number_of_paths_to_output(G):
    """
    Computes the relative number of paths to the output.
    """
    paths, _ = number_of_paths_to_each_node(G)
    summ = np.sum(list(paths.values())) - zero_degree
    return paths[-1] / summ


def path_distribution(G, posG, stages, stage=None):
    """
    For each node computes the path length distribution.
    """
    Graph = G if stage is None else gu.get_subgraph(G, posG, stages, stage)
    sort = nx.topological_sort(Graph)
    size = len(Graph.nodes())
    paths = {i: [0] * (size - 1) for i in Graph.nodes()}
    # paths = np.zeros((size,size-1))
    in_degree = Graph.in_degree()
    zero_degree = 0
    for i, n in enumerate(sort):
        if in_degree[n] == 0:
            paths[n][0] = 1
            paths[n] = np.array(paths[n])
            zero_degree += 1
        for neighbour in Graph[n]:
            rolled = np.roll(paths[n], 1)
            rolled[0] = 0
            paths[neighbour] += rolled
    return paths, zero_degree

import os

import networkx as nx

from src.models import graph_utils
from src.models.composite import composite
from src.models.random_dag import random_dag
from src.models.straight_line import straight_line


def ver(s, k):
    return s + '_v' + str(k)


def writeout(name, content, net_dir='graphs', overwrite=False):
    """
    saves the architecture defined in 'content' string to file with 'name' in directory 'net_dir'.
    :param name: the name of the file.
    :param content: the content to save.
    :param net_dir: the directory where the content should be saved.
    :param overwrite: whether to overwrite.
    :return:
    """
    fname = os.path.join(net_dir, name + '.py')
    if overwrite or not os.path.isfile(fname):
        print('writing', name)
        with open(fname, 'w') as f:
            f.write(content)
    else:
        print(name, 'exists, skipping...')


def generate_nx(**kwargs):
    """
    Generate the networkx graph.
    :param kwargs: the graph parameters (see paper).
    :return:
    """
    graph_type = kwargs['type']
    n = kwargs['n']
    seed1 = kwargs['seed1']
    seed2 = kwargs.get('seed2', 0)
    emb = kwargs.get('emb', 'spring')
    if graph_type in ['er', 'ba', 'ws']:

        if graph_type == 'er':
            G = nx.erdos_renyi_graph(n, kwargs['p'], seed=seed1)

        if graph_type == 'ba':
            G = nx.barabasi_albert_graph(n, kwargs['m'], seed=seed1)

        if graph_type == 'ws':
            G = nx.connected_watts_strogatz_graph(n, kwargs['k'], kwargs['p'], seed=seed1)

        if emb == 'spring':
            posG = nx.drawing.spring_layout(G, seed=seed2)
        if emb == 'kamada':
            posG = nx.drawing.kamada_kawai_layout(G)

        dag = kwargs['dag']
        if dag == 'x':
            H, posH, rH = graph_utils.make_dag(G, posG)
        if dag == 'radial':
            H, posH, rH = graph_utils.make_dag(G, posG, dag_ordering=graph_utils.dag_radial)
        if dag == 'radialrev':
            H, posH, rH = graph_utils.make_dag(G, posG, dag_ordering=graph_utils.dag_radial)
            H, posH, rH = graph_utils.reverse(H, posH, rH)
        if dag == 'bifocal':
            H, posH, rH = graph_utils.make_dag(G, posG, dag_ordering=graph_utils.dag_bifocal)

        graph_utils.fix_orphans(H)
        return H, posH

    elif graph_type == 'random_dag':
        nout_dist = kwargs['dist']
        nout_par = kwargs['par']
        B = kwargs.get('B', 5)
        alpha = kwargs.get('alpha', 0.5)
        func = kwargs.get('func', 'exp2')

        G = random_dag(n, nout_dist=nout_dist, nout_par=nout_par, B=B, alpha=alpha, func=func, seed=seed1)
        posG = nx.drawing.kamada_kawai_layout(G)  # important only for plotting...
        return G, posG
    elif graph_type == 'composite':
        p = kwargs['p']
        return composite(n, p=p, seed=seed1)
    elif graph_type == 'fmri':
        from src.models.fmri import fmri
        threshold = kwargs['threshold']
        G = fmri(n, threshold, seed1)
        # print(G.number_of_nodes(), G.nodes())
        posG = nx.drawing.kamada_kawai_layout(G)
        H, posH, rH = graph_utils.make_dag(G, posG)
        graph_utils.fix_orphans(H)
        return H, posH
    elif graph_type == "straight_line":
        G = straight_line(n)
        posG = nx.drawing.kamada_kawai_layout(G)
        return G, posG
    else:
        raise ValueError("Unknown graph type {}".format(graph_type))


def generate(**kwargs):
    """
    First generates the networkx graph, and then the pytorch code corresponding to the neural network defined on that graph.
    :param kwargs: graph parameters (see paper)
    :return:
    """
    H, posH = generate_nx(**kwargs)
    return graph_utils.pytorch_code(H, posH, kwargs['stages'], meta=kwargs)

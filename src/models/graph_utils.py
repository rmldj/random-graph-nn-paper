import itertools

import networkx as nx
import numpy as np


def get_subgraph(H, posH, stages, nr):
    """
    get the subgraph induced by the given stage
    :param H: the graph.
    :param posH: the nodes positions.
    :param stages: the node stages array.
    :param nr: the number of the stage in interest.
    :return:
    """
    num_nodes = H.number_of_nodes()
    stage = get_stage_dict(num_nodes, stages)
    sub_nodes = []
    for i in stage:
        if stage[i] == nr:
            sub_nodes.append(i)
    sG = H.subgraph(sub_nodes)
    return sG


def maxdistance(paths):
    """
    Returns the longest out of the distances between nodes
    paths is list(nx.all_pairs_shortest_path(er))
    """
    maxdist = 0
    for k in paths:
        lastnode = list(k[1])[-1]
        dist = len(k[1][lastnode])
        if dist > maxdist:
            maxdist = dist
    return maxdist


def best_nodes(G, debug=False):
    """
    Select two nodes which have the longest minimal path between them
    In case there are several pairs at the same distance, choose the pair
    which has the largest geometric mean of degrees.
    In dag_bifocal, these nodes will serve as the input and output node of the network
    """
    paths = list(nx.all_pairs_shortest_path(G))
    maxdist = maxdistance(paths)
    maxscore = 0
    bestnodes = []
    for k in paths:
        nodei = k[0]
        mult_i = len(G[nodei])
        nodes = list(k[1])
        for n in nodes:
            if len(k[1][n]) < maxdist:
                continue
            mult_j = len(G[n])
            score = np.sqrt(mult_i * mult_j)
            if debug:
                print(nodei, mult_i, n, mult_j, maxdist, score)
            if score > maxscore:
                maxscore = score
                bestnodes = (nodei, n)
    return bestnodes


# graph -> Directed Acyclic Graph transformations

def norm(arr):
    return np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2)


def dag_x(G, posarr):
    return posarr[:, 0]


def dag_y(G, posarr):
    return posarr[:, 1]


def dag_radial(G, posarr):
    return norm(posarr)


def dag_bifocal(G, posarr):
    n1, n2 = best_nodes(G)
    d1 = norm(posarr - posarr[n1])
    d2 = norm(posarr - posarr[n2])
    return (d1 - d2) / (d1 + d2)


def make_dag(G, pos, dag_ordering=dag_x):
    """
    Make a Directed Acyclic Graph out of graph G
    Uses a graph embedding e.g. pos = nx.drawing.spring_layout(G, seed=0)
    and a global ordering function dag_ordering.

    Use the returned positions posH for plotting the graph
    """

    # change position dictionary to a numpy array
    nodes = sorted(list(pos))
    num_nodes = len(nodes)
    posarr = np.zeros((num_nodes, 2))
    for n in nodes:
        posarr[n, :] = pos[n]

    # nodes will be ordered according to increasing r
    r = dag_ordering(G, posarr)
    ind = np.argsort(r)
    G2 = nx.relabel_nodes(G, {ind[i]: i for i in range(num_nodes)})

    # construct directed graph

    H = nx.DiGraph()
    H.add_nodes_from(G2)
    H.add_edges_from([(min(i, j), max(i, j)) for (i, j) in G2.edges])

    posH = {i: pos[ind[i]] for i in range(num_nodes)}
    rH = r[ind]

    return H, posH, rH


def fix_orphans(H, debug=False):
    """
    Add edges to nodes in a DAG, which have either all ingoing edges or all outgoing edges
    """
    num_nodes = H.number_of_nodes()
    edges_added = 0
    for n in range(num_nodes):
        incoming_edges = list(H.predecessors(n))
        outgoing_edges = list(H.successors(n))
        if n > 0 and len(incoming_edges) == 0:
            H.add_edge(n - 1, n)
            edges_added += 1
            if debug:
                print('added edge', n - 1, n)
        if n < num_nodes - 1 and len(outgoing_edges) == 0:
            H.add_edge(n, n + 1)
            edges_added += 1
            if debug:
                print('added edge', n, n + 1)


def reverse(H, posH, rH):
    """
    Computes the reversed dag orientation.
    """
    Hrev = H.reverse()
    # relabel nodes in order to agree with our conventions
    num_nodes = H.number_of_nodes()
    Hrev = nx.relabel_nodes(Hrev, {i: num_nodes - i - 1 for i in range(num_nodes)})
    posHrev = {i: posH[num_nodes - i - 1] for i in range(num_nodes)}
    rHrev = rH[::-1]
    return Hrev, posHrev, rHrev


# separate the nodes into distinct stages dealing with image of given size
# in case of CIFAR-10, there will be three stages labeled by 0, 1, 2 corresponding
# to images of size 32x32, 16x16 and 8x8 respectively

def get_stage_dict(num_nodes, stages):
    '''
    returns a dict which specifies for each node to which stage (image size) it belongs
    stages is a list/tuple specifying the number of nodes in each stage (apart from the final one)
    the remaining nodes are put to the final stage
    
    get_stage_dict(6,[2,3])
    {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 2}
    '''
    breakpoints = list(itertools.accumulate(stages))
    ranges = ([range(breakpoints[0])] + [range(breakpoints[i], breakpoints[i + 1]) for i in
                                         range(len(breakpoints) - 1)] +
              [range(breakpoints[-1], num_nodes)])
    stage = dict()
    for s, r in enumerate(ranges):
        for i in r:
            stage[i] = s
    return stage


# colors of nodes belonging to different stages
stagecolor = {0: 'c', 1: 'g', 2: 'm'}


# drawing routine

def draw_dag(H, posH, stages, nr=None):
    """
    Draws a DAG incorporating info about stages.
    Nodes are coloured according to stages.
    Light edges between nodes in different stages (need reduction).
    """
    if stages is not None:
        num_nodes = H.number_of_nodes()
        stage = get_stage_dict(num_nodes, stages)
        node_colors = [stagecolor[stage[n]] for n in H.nodes()]
        edge_colors = [('k' if stage[i] == stage[j] else 'burlywood') for (i, j) in H.edges]
    else:
        num_nodes = H.number_of_nodes()
        nr = nr if nr is not None else 0
        node_colors = [stagecolor[nr] for n in H.nodes()]
        edge_colors = ['k' for (i, j) in H.edges]
    nx.draw(H, posH, with_labels=True, node_color=node_colors, edge_color=edge_colors)


# utilities for generating PyTorch code corresponding to the graph (DAG)

class Recorder:

    def __init__(self, tab='        '):
        self.tab = tab
        self.rec = ''

    def append(self, s):
        self.rec += self.tab + s + '\n'


def multiplier(prefix, cf):
    if cf == 0:
        return ''
    else:
        return '{}{}'.format(prefix, 2 ** cf)


def network(H, stages, num_outputs=10):
    """
    Generates the body of the constructor and the body of self.forward(x)
    """
    num_nodes = H.number_of_nodes()
    num_units = 0  # number of nodes + Reduce units
    edges = []  # this will be built up successively in the same order as the edge_weights(self) function
    stage = get_stage_dict(num_nodes, stages)
    init = Recorder()
    forward = Recorder()
    params = Recorder()
    edge_weights = Recorder()

    params.append("C = sym.symbols('C')")
    edge_weights.append('lst = []   # list holding edge weights in the same order as edges')

    for n in range(num_nodes):
        incoming = sorted(list(H.predecessors(n)))
        outgoing = sorted(list(H.successors(n)))
        reductions = set([stage[m] - stage[n] for m in outgoing])
        # need_reduction = [m for m in outgoing if stage[m]!=stage[n]]
        if n == 0:
            init.append('self.n0   = Input(C, restype=self.restype, blocktype=self.blocktype)')
            forward.append('x0   = self.n0(x)')
            params.append('total = Input.params_count_sym(3,C,restype=restype, blocktype=blocktype)')
            num_units += 1
        else:
            init.append('self.n{}   = Node(C{}, {}, restype=self.restype, blocktype=self.blocktype)'.format(
                n,
                multiplier('*', stage[n]),
                len(incoming)))

            num_units += 1
            partial = 'x{}   = self.n{}('.format(n, n)
            if len(incoming) == 1:
                i = incoming[0]
                edges.append((i, n))
                edge_weights.append('lst.append(1.0)      # {}'.format((i, n)))
                reduction = stage[n] - stage[i]
                partial += 'x{}{})'.format(
                    i,
                    multiplier('_', reduction))
            else:
                partial += '['
                for index, i in enumerate(incoming):
                    edges.append((i, n))
                    edge_weights.append('lst.append(torch.sigmoid(self.n{}.weights[{}]).item())    # {}'.format(
                        n,
                        index,
                        (i, n)))

                    reduction = stage[n] - stage[i]
                    partial += ' x{}{},'.format(
                        i,
                        multiplier('_', reduction))
                partial += '])'
            forward.append(partial)
            params.append('total += Node.params_count_sym(C{}, C{}, {}, restype=restype, blocktype=blocktype )'.format(
                multiplier('*', stage[n]),
                multiplier('*', stage[n]),
                len(incoming)))

        for reduction in reductions:
            if reduction == 0:
                continue
            init.append('self.r{}{} = Reduce(C{}, C{}, {}, restype=self.restype, blocktype=self.blocktype)'.format(
                n,
                multiplier('x', reduction),
                multiplier('*', stage[n]),
                multiplier('*', stage[n] + reduction),
                2 ** reduction))

            num_units += 1
            forward.append('x{0}{1} = self.r{0}{2}(x{0})'.format(
                n,
                multiplier('_', reduction),
                multiplier('x', reduction)))

            params.append('total += Reduce.params_count_sym(C{}, C{}, restype=restype, blocktype=blocktype)'.format(
                multiplier('*', stage[n]),
                multiplier('*', stage[n] + reduction)))

    init.append('self.out   = Output(C{}, num_outputs)'.format(multiplier('*', stage[n])))
    forward.append('return self.out(x{})'.format(n))
    params.append('total += Output.params_count_sym(C{}, {})'.format(
        multiplier('*', stage[n]),
        num_outputs))

    params.append('return total, C')
    edge_weights.append('return lst')
    return num_units, edges, init.rec, forward.rec, params.rec, edge_weights.rec


def pytorch_code(H, posH, stages, meta=None):
    num_units, edges, init, fwd, params, edge_weights = network(H, stages)

    OUTPUT = '''
import torch
import torch.nn as nn
from src.elementary_modules import Input, Node, Reduce, Output
from numpy import array
import sympy as sym

class Net(nn.Module):
    
    # Information about the original graph               
    meta = {meta}
    stages = {stages}
    num_nodes = {num_nodes}
    num_units = {num_units}
    edges = {edges}
    pos = {pos}
    
    
    def __init__(self, C, restype="C", blocktype="simple", num_outputs=10):
        super().__init__()

        self.restype = restype
        self.blocktype = blocktype
        
        # Neural Network proper
'''.format(meta=meta, stages=stages, num_nodes=H.number_of_nodes(), num_units=num_units, edges=edges, pos=posH)

    OUTPUT += init

    OUTPUT += '''
    
    def forward(self, x):
'''

    OUTPUT += fwd

    OUTPUT += '''

    def edge_weights(self):    
'''
    OUTPUT += edge_weights

    OUTPUT += '''
    
    @staticmethod
    def params_count_sym(restype="C", blocktype="simple"):
'''
    OUTPUT += params

    OUTPUT += '''

    @classmethod
    def get_C(cls, params_count, restype, blocktype):
        total, C = cls.params_count_sym(restype, blocktype)
        sol = sym.solve([total-params_count,C>=0], C)
        return round(sym.solve(sol)[0].evalf())
        
        
    #------------- TESTS -------------#
        
def test():
        test_static()
        for i in [8, 16, 32, 64]:
            net = Net(i)
            test_parameters(net, i)
        
def test_parameters(net, c):
        import numpy as np
        total_params = 0

        for x in filter(lambda p: p.requires_grad, net.parameters()):
            total_params += np.prod(x.data.numpy().shape)
        print("Total number of params", total_params)
        total, C = Net.params_count_sym()
        assert total_params == total.subs(C, c), "symbolic computations do not match true parameters count"
        
def test_static():
        print(Net.meta, Net.stages, Net.num_nodes,
              Net.num_units, Net.edges, Net.pos)

'''

    return OUTPUT

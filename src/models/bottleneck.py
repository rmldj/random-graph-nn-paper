import argparse
import importlib
import sys

import networkx as nx

from src.models import graph_utils
from src.models.generate import writeout


def get_parser():
    parser = argparse.ArgumentParser(description='bottleneck preparator')
    parser.add_argument('arch', metavar='ARCH', type=str,
                        help='model net from --net-dir')
    parser.add_argument('--net-dir', dest='net_dir',
                        help='The directory with the network model definitions (default: graphs)',
                        default='graphs', type=str)
    parser.add_argument('--results-dir', dest='results_dir',
                        help='The directory used to save simulation results (default: graphs)',
                        default='graphs', type=str)
    parser.add_argument('--nr-stages', type=int, default=3, help='number of stages in network')
    return parser


def make_bottleneck(args, Net):
    """
    makes a bottleneck graph.
    :param args: command line arguments.
    :param Net: the network which to transform to its bottleneck variant.
    :return:
    """
    print('getting the graph...')
    H = get_graph(Net)

    print('cutting edges...')
    stage_dict = graph_utils.get_stage_dict(Net.num_nodes, Net.stages)
    edges_interstage = [e for e in Net.edges if stage_dict[e[0]] != stage_dict[e[1]]]
    H.remove_edges_from(edges_interstage)

    if args.nr_stages == 3:
        n0 = Net.stages[0]
        n1 = Net.stages[1]

        H.add_edge(n0 - 1, n0)
        H.add_edge(n1 - 1, n1)
        print('Fixing orphans...')
        graph_utils.fix_orphans(H)
    else:
        raise NotImplementedError("not implemented for more or less than 3 stages")

    return graph_utils.pytorch_code(H, Net.pos, Net.stages, Net.meta)


def get_graph(Net):
    H = nx.DiGraph()
    H.add_nodes_from(range(Net.num_nodes))
    H.add_edges_from(Net.edges)
    return H


def main(args):
    print('loading Net')
    sys.path.append(args.net_dir)
    Net = getattr(importlib.import_module(args.arch), 'Net')
    print('preparing content...')
    content = make_bottleneck(args, Net)
    print('writing...')
    writeout("bottleneck_{}".format(args.arch), content)
    print('done.')


if __name__ == "__main__":
    print('getting parser...')
    parser = get_parser()
    print('getting arguments...')
    args = parser.parse_args()
    main(args)

import argparse
import importlib
import sys
from pprint import pprint

import matplotlib.pyplot as plt
import networkx as nx

from src.models import graph_utils


def get_parser():
    parser = argparse.ArgumentParser(description='Simulations on CIFAR10')
    parser.add_argument('arch', metavar='ARCH', help='net model from --net-dir', type=str)
    parser.add_argument('--net-dir', dest='net_dir',
                        help='The directory with the network model definitions (default: graphs)', default='graphs',
                        type=str)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--save-path', default='.', help='path where to save the images')
    return parser


def main(args):
    sys.path.append(args.net_dir)
    Net = getattr(importlib.import_module(args.arch), 'Net')

    net = Net(4)
    pprint(net.meta)
    print()
    print('nodes', net.num_nodes)
    print('units', net.num_units)
    print('C(small) :', Net.get_C(464154, restype="C", blocktype="simple"))
    print('C(medium):', Net.get_C(853018, restype="C", blocktype="simple"))
    print('C(large) :', Net.get_C(1727962, restype="C", blocktype="simple"))

    H = nx.DiGraph()
    H.add_nodes_from(range(net.num_nodes))
    H.add_edges_from(net.edges)
    plt.figure(figsize=(9, 7))
    graph_utils.draw_dag(H, net.pos, net.stages)
    if args.save:
        plt.savefig('{}/{}.png'.format(args.save_path, args.arch), bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)

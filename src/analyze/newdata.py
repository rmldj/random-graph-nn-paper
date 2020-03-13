import math
import numpy as np
import pandas as pd
import sys
import importlib
from sklearn.decomposition import PCA
import networkx as nx
import pickle
from src.analyze import analyze_utils as au
import argparse
import os
from time import time


def get_parser():
    parser = argparse.ArgumentParser(description='Compute the n_bottleneck and pca_elongation features.')
    parser.add_argument('--net-dir', dest='net_dir',
                        help='The directory with the network model definitions (default: ./graphs)',
                        default='./graphs', type=str)
    parser.add_argument('--output',
                        help='The directory used to save features (default: ./reports/newdata<nr-nodes>.pkl)',
                        default='./reports', type=str)
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    y30, y30t, y60, y60t = load_data()

    new_train_30 = get_df_for_graphs(y30.index, args)
    new_test_30 = get_df_for_graphs(y30t.index, args)

    new_data_30 = {'X_train': new_train_30, 'X_test': new_test_30}

    print()

    new_train_60 = get_df_for_graphs(y60.index, args)
    new_test_60 = get_df_for_graphs(y60t.index, args)

    new_data_60 = {'X_train': new_train_60, 'X_test': new_test_60}

    with open(os.path.join(args.output,"newdata30.pkl"), "wb") as file:
        pickle.dump(new_data_30, file)
    with open(os.path.join(args.output, "newdata60.pkl"), "wb") as file:
        pickle.dump(new_data_60, file)


def load_data():
    Xall30 = pd.read_pickle('./reports/data30.pkl')
    y30 = Xall30['y_train']
    y30t = Xall30['y_test']
    Xall60 = pd.read_pickle('./reports/data60.pkl')
    y60 = Xall60['y_train']
    y60t = Xall60['y_test']
    return y30, y30t, y60, y60t


def select_best(H, pos):

    adj = nx.to_numpy_array(H)
    num_nodes = H.number_of_nodes()
    adjcum = np.zeros_like(adj)

    res = dict()

    for i in range(num_nodes-1):
        row = adj[i]
        adjcum[i, i+1:] = (np.cumsum(row[::-1])[::-1])[i+1:]

    widths = np.sum(adjcum, axis=0)
    n_bottlenecks = np.sum(widths == 1)

    res['n_bottlenecks'] = n_bottlenecks

    positions = np.zeros((num_nodes,2))
    for i in range(num_nodes):
        positions[i, :] = pos[i]

    pca = PCA()
    pca.fit(positions)
    pca_length = pca.explained_variance_ratio_[0]
    res['pca_elongation'] = 2 * (pca_length - 0.5)

    return res


def get_df_for_graphs(graphs, args):

    ngraphs = len(graphs)

    df = None
    t0 = time()

    for i, g in enumerate(graphs):
        print('{}/{} {}'.format(i + 1, ngraphs, g))
        info = au.graphinfo(g, args.net_dir)

        pos = info['pos']
        H = info['G']

        features = select_best(H, pos)

        if df is None:
            columns = features.keys()
            df = pd.DataFrame(index=graphs, columns=columns)
        df.loc[g] = features
        print(features)
        print()
    print('finished in', time() - t0)
    return df


if __name__ == "__main__":
    main()

    






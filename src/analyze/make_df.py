import argparse
from time import time

import pandas as pd

from src.analyze import analyze_utils as au


def get_parser():
    parser = argparse.ArgumentParser(description='Netwrok analysis on the models architecture.')
    parser.add_argument('--input-results', default='./reports/dfresults.pkl')
    parser.add_argument('--net-dir', dest='net_dir',
                        help='The directory with the network model definitions (default: ./graphs)',
                        default='./graphs', type=str)
    parser.add_argument('--output',
                        help='The directory used to save features (default: ./reports/dffeaturestotal.pk)',
                        default='./reports/dffeaturestotal.pkl', type=str)
    return parser


def make_features(results_path, output_path):
    """
    Computes the features for all graphs, which are present in the results dataframe.
    :param results_path: The results dataframe.
    :param output_path: The output path.
    :return:
    """
    results = pd.read_pickle(results_path)
    graphs = results.index.to_list()
    ngraphs = len(graphs)

    df = None
    t0 = time()

    for i, g in enumerate(graphs):
        print('{}/{} {}'.format(i + 1, ngraphs, g))
        info = au.graphinfo(g, net_dir=args.net_dir)
        features0 = au.features_part0(info)
        features1 = au.features_part1(info)
        features2 = au.features_part2(info)
        features = {**features0, **features1, **features2}
        columns = features.keys()
        if df is None:
            df = pd.DataFrame(index=graphs, columns=columns)
        df.loc[g] = features
        print(features)
        print()
    print('finished in', time() - t0)
    df.to_pickle(output_path)


def main():
    parser = get_parser()
    args = parser.parse_args()
    make_features(args.input_results, args.output)


if __name__ == "__main__":
    main()

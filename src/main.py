import argparse
from src.train.simulate import simulate

def get_parser():
    parser = argparse.ArgumentParser(description='Simulations on CIFAR10')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                        help='model architecture: either resnet32|56|110 or net from --net-dir (default: resnet32)')
    parser.add_argument('--seed', default=1621, type=int, metavar='SEED',
                        help='random seed (default: 1621)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run (default: 100)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate (default: 0.1)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--half', dest='half', action='store_true',
                        help='use half-precision(16-bit) ')
    parser.add_argument('--deterministic', dest='deterministic', action='store_true',
                        help='deterministic cudnn setting (but different results for different GPUs) ')
    parser.add_argument('--milestone1', default=80, type=int, metavar='EPOCH',
                        help='reduce learning rate 1 (default: 80)')
    parser.add_argument('--milestone2', default=90, type=int, metavar='EPOCH',
                        help='reduce learning rate 2 (default: 90)')
    parser.add_argument('--save-model', dest='save_model', action='store_true',
                        help='save model after training')
    parser.add_argument('--save-preds', dest='save_preds', action='store_true',
                        help='save final test set predictions')
    parser.add_argument('--net-dir', dest='net_dir',
                        help='The directory with the network model definitions (default: graphs)',
                        default='graphs', type=str)
    parser.add_argument('--results-dir', dest='results_dir',
                        help='The directory used to save simulation results (default: results)',
                        default='results', type=str)
    parser.add_argument('--models-dir', dest='models_dir',
                        help='The directory used to save trained models if --save_model is set (default: models)',
                        default='models', type=str)
    parser.add_argument('--preds-dir', dest='preds_dir',
                        help='The directory used to save final test set predictions if --save_preds is set (default: preds)',
                        default='preds', type=str)
    parser.add_argument('--C', type=int, metavar='C',
                        help='If set, fixes the number of out channels in the first network layer')
    parser.add_argument('--size', dest='size', choices=['L', 'M', 'S'], default='M',
                        help='Size of the model L:resnet110 like, M:resnet56 like, S:resnet32 like (default: M)')
    parser.add_argument('--verbose', action="store_true", help="turns on the verbose mode, when losses, accuracy"
                                                               " and learning-rates are printed during learning")

    parser.add_argument('--restype', choices=["A", "B", "C", "None"], default="C",
                        help="type of the residual connection. "
                             "Naming convention as in ResNet paper")

    parser.add_argument('--blocktype', choices=["res", "simple"], default="simple",
                        help="type of the computation block in each node. "
                             "Use 'simple' for sum-relu-conv-batchnorm. "
                             "Use 'res' for ResNet unit")

    parser.add_argument('--cifar', choices=["10", "100"], default="10",
                        help="'10' for CIFAR10 (default) '100' for CIFAR100")
    parser.add_argument('--datapath', default="./data", help="the path for the data")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    simulate(args)


if __name__ == '__main__':
    main()

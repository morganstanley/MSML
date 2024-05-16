from argparse import ArgumentParser
import distutils.util


# import torch


def get_flags():
    parser = ArgumentParser(description='Argument Parser')

    parser.add_argument("--seed", default=3407, type=int, help="Random seed")

    parser.add_argument("--optimizer", default="sgld", type=str, help="Optimizer")

    parser.add_argument("--lr", default=5e-6, type=float, help="Learning rate")

    # parser.add_argument("--dims", default=2, type=int, help="Parameter dimensions")
    parser.add_argument("--n_epoch", default=6e4, type=int, help="Number of epochs")
    parser.add_argument("--batch", default=4096, type=int, help="Batch size")
    parser.add_argument("--decay_rate", default=0.9999, type=float, help="Decay rate")


    # parser.add_argument("--save_after", default=10, type=int, help="Save samples after number of steps")
    # parser.add_argument("--plot_after", default=1000, type=int, help="Save plots after number of steps")
    # parser.add_argument("--metric_after", default=100, type=int, help="Calculate metrics after number of steps")

    # parser.add_argument("--lower", default=-2.5, type=float, help="grid lower bound")
    # parser.add_argument("--upper", default=2.5, type=float, help="grid upper bound")

    parser.add_argument("--T_low", default=1, type=float, help="grid lower bound")
    parser.add_argument("--T_high", default=10, type=float, help="grid lower bound")
    parser.add_argument("--hat_var", default=8.6, type=float, help="grid lower bound")
    parser.add_argument("--threshold", default=3e-4, type=float, help="grid lower bound")

    parser.add_argument("--warm_up", default=1000, type=int, help="Number of homotopy steps")

    parser.add_argument("--num_points", default=200, type=int, help="Number of points (for grid)")
    parser.add_argument("--num_samples", default=100, type=int, help="Number of points (for svgd)")

    parser.add_argument("--path", default="./logs/figs/SGLD_result/", type=str, help="Directory for figures")

    parser.add_argument('--if_include_domain', default=True, type=lambda x: bool(distutils.util.strtobool(x)),
                        help='Calculate kernel stein discrepancy')
    parser.add_argument('--if_save_metrics', default=True, type=lambda x: bool(distutils.util.strtobool(x)),
                        help='save samples and kl-divergence')

    parser.add_argument('--if_gifs', default=True, type=lambda x: bool(distutils.util.strtobool(x)),
                        help='save samples and kl-divergence')

    args = parser.parse_args()

    return args

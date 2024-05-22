from argparse import ArgumentParser
import distutils.util


# import torch


def get_flags():
    parser = ArgumentParser(description='Argument Parser')

    parser.add_argument("--seed", default=820, type=int, help="Random seed")

    parser.add_argument("--optimizer", default="sgld", type=str, help="Optimizer")
    # parser.add_argument("--optimizer", default="svgd", type=str, help="Optimizer")
    # parser.add_argument("--optimizer", default="cyclic_sgld", type=str, help="Optimizer")
    # parser.add_argument("--optimizer", default="csgld", type=str, help="Optimizer")

    parser.add_argument("--domain_type", default="flower", type=str, help="Optimizer")
    parser.add_argument("--radius", default=2, type=float, help="Learning rate 2")

    parser.add_argument("--lr", default=3e-3, type=float, help="Learning rate 2")
    parser.add_argument("--regularization", default=None, type=float, help="L2 regularization")
    parser.add_argument("--n_epoch", default=1e5, type=int, help="Number of samples")

    parser.add_argument("--dims", default=2, type=int, help="parameter dimensions")
    parser.add_argument("--M", default=20, type=int, help="Number of period in cyclic SGLD")
    parser.add_argument("--T_low", default=1.0, type=float, help="Low temp for reSGLD")
    parser.add_argument("--T_high", default=10.0, type=float, help="High temp for reSGLD")
    parser.add_argument("--threshold", default=3e-4, type=float, help="swap threshold for reSGLD")
    parser.add_argument("--lr_gap", default=2.0, type=float, help="lr gap for reSGLD")
    parser.add_argument("--hat_var", default=8.0, type=float, help="hat_var for reSGLD")
    parser.add_argument("--warm_up", default=100000, type=int, help="burn-in samples")

    parser.add_argument("--decay_rate", default=3e-3, type=float, help="Decay rate")
    parser.add_argument("--zeta", default=1.0, type=float, help="zeta for csgld")


    parser.add_argument("--save_after", default=10, type=int, help="Save samples after number of steps")
    parser.add_argument("--plot_after", default=1000, type=int, help="Save plots after number of steps")
    parser.add_argument("--metric_after", default=100, type=int, help="Calculate metrics after number of steps")

    parser.add_argument("--lower", default=-2.5, type=float, help="grid lower bound")
    parser.add_argument("--upper", default=2.5, type=float, help="grid upper bound")


    parser.add_argument("--num_points", default=100, type=int, help="Number of points (for grid)")

    parser.add_argument("--path", default="./logs/figs/SGLD_result/", type=str, help="Directory for figures")

    parser.add_argument('--if_include_domain', default=True, type=lambda x: bool(distutils.util.strtobool(x)),
                        help='Calculate kernel stein discrepancy')
    parser.add_argument('--if_save_metrics', default=True, type=lambda x: bool(distutils.util.strtobool(x)),
                        help='save samples and kl-divergence')

    parser.add_argument('--if_gifs', default=True, type=lambda x: bool(distutils.util.strtobool(x)),
                        help='save samples and kl-divergence')
                        
                        

    args = parser.parse_args()

    return args

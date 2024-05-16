from argparse import ArgumentParser


def get_flags():
    parser = ArgumentParser(description='Argument Parser')

    # ------------------------------------ Hyper parameters ------------------------------------
    parser.add_argument("--seed", default=3407, type=int, help="Set random seed")

    parser.add_argument("--dim", default=10, type=int, help="Parameter dimensions")
    parser.add_argument("--n_arm", default=10, type=int, help="Number of arms")
    parser.add_argument("--n_round", default=1000, type=int, help="Number of rounds")
    parser.add_argument("--n_iter", default=10, type=int, help="Number of iterations each round")
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size for stochastic gradient estimate")

    parser.add_argument("--gamma", default=2.0, type=float, help="Friction coefficient")
    parser.add_argument("--step_size", default=1e-4, type=float, help="Step size")

    return parser.parse_args()

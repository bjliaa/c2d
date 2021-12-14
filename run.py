import argparse
from c2d.runner import Runner

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run an experiment for STEPS total environment steps.")
    parser.add_argument(
        "--game",
        dest="game",
        default="breakout",
        help="Environment string. Default is breakout.",
    )
    parser.add_argument(
        "--tag",
        dest="dtag",
        default="unnamed",
        help="Training run data string tag. Default is no data storage.",
    )
    parser.add_argument(
        "--iterations",
        dest="iterations",
        type=int,
        default=1,
        help="Number of training and evaluation phases. Default is 1 iterations.",
    )
    args = parser.parse_args()
    runner = Runner(args.game, args.iterations, args.dtag)
    runner.run()

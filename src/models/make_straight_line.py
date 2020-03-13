import os

from src.models.generate import generate, writeout, ver


def main():
    """
    creates the chain-like architectures.
    """
    os.makedirs("graphs", exist_ok=True)

    SEEDS = [1621, 555, 137, 331, 2884]

    n = 30
    stages = [10, 10]

    for i, seed in enumerate(SEEDS):
        writeout(ver('straight_line_{}'.format(n), i),
                 generate(type='straight_line', n=n, stages=stages, seed1=seed))

    n = 60
    stages = [20, 20]

    for i, seed in enumerate(SEEDS):
        writeout(ver('straight_line_{}'.format(n), i),
                 generate(type='straight_line', n=n, stages=stages, seed1=seed))


if __name__ == "__main__":
    main()

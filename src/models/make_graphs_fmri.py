import os

from src.models.generate import generate, writeout, ver


def main():
    """
    Creates the fMRI architectures.
    """
    os.makedirs("graphs", exist_ok=True)

    SEEDS = [1621, 555, 137, 331, 2884]

    n = 30
    stages = [10, 10]

    thresholds = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 4.9]

    for threshold in thresholds:
        for i, seed in enumerate(SEEDS):
            writeout(ver('fmri_{}_{}'.format(int(threshold * 10), n), i),
                     generate(type='fmri', n=n, threshold=threshold, stages=stages, seed1=seed))

    n = 60
    stages = [20, 20]

    thresholds = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    for threshold in thresholds:
        for i, seed in enumerate(SEEDS):
            writeout(ver('fmri_{}_{}'.format(int(threshold * 10), n), i),
                     generate(type='fmri', n=n, threshold=threshold, stages=stages, seed1=seed))


if __name__ == "__main__":
    main()

import os

from src.models.generate import generate, writeout, ver


def main():
    """
    Creates the er, ws, ba, rdag and composite architectures.
    """

    os.makedirs("graphs", exist_ok=True)

    SEEDS = [1621, 555, 137, 331, 2884]

    #----------------------- ER 30 -----------------------

    n0 = 30
    stages = [10, 10]
    p0s = {0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0}

    for p in p0s:
        for k, seed in enumerate(SEEDS):
            writeout(ver('er1kx_{}_{}'.format(int(p * 100), n0), k), generate(type='er', n=n0, p=p, stages=stages,
                                                                              seed1=seed, dag='x', emb='kamada'))
    #----------------------- ER 60 -----------------------

    n1 = 60
    stages1 = [20, 20]
    p1s = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
    for p in p1s:
        for k, seed in enumerate(SEEDS):
            writeout(ver('er1kx_{}_{}'.format(int(p * 100), n1), k),
                     generate(type='er', n=n1, p=p, stages=stages1, seed1=seed, dag='x', emb='kamada'))

    #----------------------- WS 30 -----------------------

    n2 = 30
    stages = [10, 10]
    p2s = {0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 1.0}
    k2s = {2, 4, 6, 8}

    for p in p2s:
        for k in k2s:
            for i, seed in enumerate(SEEDS):
                writeout(ver('wskx_{}_{}_{}'.format(int(p * 100), n2, k), i),
                         generate(type='ws', n=n2, k=k, p=p, stages=stages, seed1=seed, dag='x', emb='kamada'))

    #----------------------- BA 30 -----------------------

    ms = {2, 3, 5, 7, 11}
    for m in ms:
        for i, seed in enumerate(SEEDS):
            writeout(ver('bakx_{}_{}'.format(m, n2), i),
                     generate(type='ba', n=n2, m=m, stages=stages, seed1=seed, dag='x', emb='kamada'))


    #----------------------- WS 60 -----------------------

    n2 = 60
    stages = [20, 20]
    p2s = {0.0, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 1.0}
    k2s = {2, 4, 6, 8, 10}

    for p in p2s:
        for k in k2s:
            for i, seed in enumerate(SEEDS):
                writeout(ver('wskx_{}_{}_{}'.format(int(p * 100), n2, k), i),
                         generate(type='ws', n=n2, k=k, p=p, stages=stages, seed1=seed, dag='x', emb='kamada'))

    #----------------------- BA 60 -----------------------

    ms = {2, 3, 5, 7, 11}
    for m in ms:
        for i, seed in enumerate(SEEDS):
            writeout(ver('bakx_{}_{}'.format(m, n2), i),
                     generate(type='ba', n=n2, m=m, stages=stages, seed1=seed, dag='x', emb='kamada'))

    #----------------------- RDAG 30 -----------------------

    n2 = 30
    stages = [10, 10]
    ms = (2, 3, 4, 5)
    for m in ms:
        for i, seed in enumerate(SEEDS):
            writeout(ver('rdag_constant_{}'.format(m), i),
                     generate(type='random_dag', n=n2, dist='constant', par=m, stages=stages, seed1=seed))
            writeout(ver('rdag_constant_{}_pow1'.format(m), i),
                     generate(type='random_dag', n=n2, dist='constant', par=m, func='pow1', stages=stages, seed1=seed))
            writeout(ver('rdag_constant_{}_exp3'.format(m), i),
                     generate(type='random_dag', n=n2, dist='constant', par=m, func='exp3', stages=stages, seed1=seed))
            writeout(ver('rdag_constant_{}_one'.format(m), i),
                     generate(type='random_dag', n=n2, dist='constant', par=m, func='one', stages=stages, seed1=seed))

    for i, seed in enumerate(SEEDS):
        writeout(ver('rdag_hubs_2_9', i),
                 generate(type='random_dag', n=n2, dist='hubs', par=(2, (1, 13, 21), 9), stages=stages, seed1=seed))
        writeout(ver('rdag_hubs_3_9', i),
                 generate(type='random_dag', n=n2, dist='hubs', par=(3, (1, 13, 21), 9), stages=stages, seed1=seed))

    for i, seed in enumerate(SEEDS):
        writeout(ver('rdag_laplace_3', i),
                 generate(type='random_dag', n=n2, dist='laplace', par=3, stages=stages, seed1=seed))


    #----------------------- RDAG 60 -----------------------

    n2 = 60
    stages2 = [20, 20]
    ms = (2, 3, 4, 5)
    for m in ms:
        for i, seed in enumerate(SEEDS):
            writeout(ver('rdag_constant_{}_{}'.format(m, n2), i),
                     generate(type='random_dag', n=n2, dist='constant', par=m, stages=stages2, seed1=seed))
            writeout(ver('rdag_constant_{}_pow1_{}'.format(m, n2), i),
                     generate(type='random_dag', n=n2, dist='constant', par=m, func='pow1', stages=stages2, seed1=seed))
            writeout(ver('rdag_constant_{}_exp3_{}'.format(m, n2), i),
                     generate(type='random_dag', n=n2, dist='constant', par=m, func='exp3', stages=stages2, seed1=seed))
            writeout(ver('rdag_constant_{}_B10_{}'.format(m, n2), i),
                     generate(type='random_dag', n=n2, dist='constant', par=m, B=10, stages=stages2, seed1=seed))
            writeout(ver('rdag_constant_{}_B10_pow1_{}'.format(m, n2), i),
                     generate(type='random_dag', n=n2, dist='constant', par=m, B=10, func='pow1', stages=stages2,
                              seed1=seed))
            writeout(ver('rdag_constant_{}_B10_exp3_{}'.format(m, n2), i),
                     generate(type='random_dag', n=n2, dist='constant', par=m, B=10, func='exp3', stages=stages2,
                              seed1=seed))

    #----------------------- COMPOSITE 30 -----------------------

    n = 30
    stages = [10, 10]
    for i, seed in enumerate(SEEDS):
        writeout(ver('composite_{}_{}'.format('85', n), i),
                 generate(type='composite', n=30, p=0.85, stages=stages, seed1=seed))
        writeout(ver('composite_{}_{}'.format('99', n), i),
                 generate(type='composite', n=30, p=0.99, stages=stages, seed1=seed))

    # ----------------------- COMPOSITE 60 -----------------------

    n = 60
    stages = [20, 20]
    for i, seed in enumerate(SEEDS):
        writeout(ver('composite_{}_{}'.format('85', n), i),
                 generate(type='composite', n=60, p=0.85, stages=stages, seed1=seed))
        writeout(ver('composite_{}_{}'.format('99', n), i),
                 generate(type='composite', n=60, p=0.99, stages=stages, seed1=seed))


if __name__ == "__main__":
    main()

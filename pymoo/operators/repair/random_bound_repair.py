import numpy as np
from random import random as rand

from pymoo.model.repair import Repair
from pymoo.util.misc import at_least_2d_array


def repair_out_of_bounds(problem, X):

    only_1d = (X.ndim == 1)
    X = at_least_2d_array(X)
    # print(X.shape)

    for i in range (len(X)):
        for j in range(len(X[i])):
            if X[i][j]<problem.xl[j] or X[i][j]>problem.xu[j]:
                X[i][j] = problem.xl[j] + rand()*(problem.xu[j]-problem.xl[j])

    # if problem.xl is not None:

        # xl = np.repeat(problem.xl[None, :], X.shape[0], axis=0)
        # X[X < xl] = xl[X < xl]

    # if problem.xu is not None:
        # xu = np.repeat(problem.xu[None, :], X.shape[0], axis=0)
        # X[X > xu] = xu[X > xu]

    if only_1d:
        return X[0, :]
    else:
        return X


class OutOfBoundsRepair(Repair):

    def _do(self, problem, pop, **kwargs):
        X = pop.get("X")
        pop.set("X", repair_out_of_bounds(problem, X))
        return pop

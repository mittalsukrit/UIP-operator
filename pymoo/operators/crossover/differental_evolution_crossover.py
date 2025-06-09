import numpy as np

from pymoo.model.crossover import Crossover
# from pymoo.operators.repair.inverse_penalty import InversePenaltyOutOfBoundsRepair
from pymoo.operators.repair.to_bound import ToBoundOutOfBoundsRepair


class DifferentialEvolutionCrossover(Crossover):

    def __init__(self, weight=0.8, dither=None, jitter=False, *args, **kwargs):
        super().__init__(3, 1, *args, **kwargs)
        self.weight = weight
        self.dither = dither
        self.jitter = jitter

    def _do(self, problem, X, **kwargs):

        n_parents, n_matings, n_var = X.shape

        if self.dither == "vector":
            weight = (self.weight + np.random.random(n_matings) * (1 - self.weight))[:, None]
        elif self.dither == "scalar":
            weight = self.weight + np.random.random() * (1 - self.weight)
        else:
            weight = self.weight

        # http://www.cs.ndsu.nodak.edu/~siludwig/Publish/papers/SSCI20141.pdf
        if self.jitter:
            gamma = 0.0001
            weight = (self.weight * (1 + gamma * (np.random.random(n_matings) - 0.5)))[:, None]

        _X = X[0] + weight * (X[1] - X[2])
        # _X = InversePenaltyOutOfBoundsRepair().do(problem, _X, P=X[0])
        _X = ToBoundOutOfBoundsRepair().do(problem, _X)

        return _X[None, ...]

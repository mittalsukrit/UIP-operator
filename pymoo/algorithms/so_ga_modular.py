import numpy as np
from scipy.spatial.distance import cdist
import warnings
from numpy.linalg import LinAlgError
from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.factory import get_decomposition
from pymoo.model.individual import Individual
from pymoo.model.population import Population
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.misc import set_if_none
from pymoo.util.normalization import normalize
from pymoo.util.function_loader import load_function
from pymoo.operators.selection.tournament_selection import TournamentSelection, compare


# =========================================================================================================
# Implementation
# =========================================================================================================

class GA(GeneticAlgorithm):

    def __init__(self,
                 display=MultiObjectiveDisplay(),
                 **kwargs):
        """

        HFIDEADN Algorithm. - normalize at every generation

        Parameters
        ----------
        ref_dirs
        display
        kwargs
        """

        set_if_none(kwargs, 'pop_size', 100)
        set_if_none(kwargs, 'sampling', FloatRandomSampling())
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob=0.9, eta=5))
        set_if_none(kwargs, 'mutation', PolynomialMutation(prob=0.1, eta=20))
        set_if_none(kwargs, 'survival', None)
        set_if_none(kwargs, 'selection', TournamentSelection(func_comp=comp_by_cv_then_random))

        super().__init__(display=display, **kwargs)

    def _initialize(self):

        super()._initialize()

    def _next(self):

        # do the mating using the current population
        self.off = self.mating.do(self.problem, self.pop, self.n_offsprings, algorithm=self)
        self.off.set("n_gen", self.n_gen)

        # set the parent and offspring attributes
        self.off.set("off", True)
        self.pop.set("off", False)

        # if the mating could not generate any new offspring (duplicate elimination might make that happen)
        if len(self.off) == 0:
            self.termination.force_termination = True
            return

        # if not the desired number of offspring could be created
        elif len(self.off) < self.n_offsprings:
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")

        # evaluate the offspring
        self.evaluator.eval(self.problem, self.off, algorithm=self)

        # merge the offsprings with the current population
        self.pop = Population.merge(self.pop, self.off)

        # the do survival selection
        self.pop = survival_selection(self)        

def survival_selection(self):
    
    # best half of parent and offspring populations
    n_survivors = self.pop_size
    X = np.array([None]*n_survivors)
    survivors = Population.new("X", X)
    set_index = 0
    pop_feas, pop_infeas = [], []
    for i in range(len(self.pop)):
        if self.pop[i].CV==0 or self.problem.n_constr==0:
            pop_feas.append(self.pop[i])
        else:
            pop_infeas.append(self.pop[i])

    # select the feasible population
    if len(pop_feas)>0:
        F_feas = np.array([ind.F[0] for ind in pop_feas])
        sorted = np.argsort(F_feas)
        for index in sorted:
            survivors[set_index].X = pop_feas[index].X
            survivors[set_index].F = pop_feas[index].F
            survivors[set_index].CV = pop_feas[index].CV
            survivors[set_index].G = pop_feas[index].G
            survivors[set_index].data = pop_feas[index].data
            survivors[set_index].set("rank", set_index)
            survivors[set_index].set("feasible", [True])
            set_index += 1
            if set_index == n_survivors:
                break
    if set_index < n_survivors:
        CV_infeas = np.array([ind.CV for ind in pop_infeas])
        sorted = np.argsort(CV_infeas)
        for index in sorted:
            survivors[set_index].X = pop_feas[index].X
            survivors[set_index].F = pop_feas[index].F
            survivors[set_index].CV = pop_feas[index].CV
            survivors[set_index].G = pop_feas[index].G
            survivors[set_index].data = pop_feas[index].data
            survivors[set_index].set("rank", set_index)
            survivors[set_index].set("feasible", [False])
            set_index += 1
            if set_index == n_survivors:
                break
    
    return survivors

def comp_by_cv_then_random(pop, P, **kwargs):
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]

        # if at least one solution is infeasible
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[i] = compare(a, pop[a].CV, b, pop[b].CV, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible just set random
        else:
            S[i] = np.random.choice([a, b])

    return S[:, None].astype(np.int)
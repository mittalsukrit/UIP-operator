import numpy as np
from scipy.spatial.distance import cdist
import math
import warnings
from numpy.linalg import LinAlgError
from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.factory import get_decomposition
from pymoo.model.individual import Individual
from pymoo.model.population import Population, pop_from_array_or_individual
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.misc import set_if_none
from pymoo.util.normalization import normalize
from pymoo.util.function_loader import load_function
from pymoo.operators.selection.tournament_selection import TournamentSelection, compare
from pymoo.operators.selection.random_selection import RandomSelection


# =========================================================================================================
# Implementation
# =========================================================================================================

class MAOEA_IGD(GeneticAlgorithm):

    def __init__(self,
                 ref_dirs,
                 n_neighbors=20,
                 decomposition='auto',
                 prob_neighbor_mating=0.9,
                 display=MultiObjectiveDisplay(),
                 **kwargs):
        """

        HFIDEAD Algorithm.

        Parameters
        ----------
        ref_dirs
        display
        kwargs

        For constraint-handling:
        add tournament selection
        divide joint pop between feasible and infeasible
        determine nadir point from feasible solutions only
        """

        self.int_gen = 1 #generations for local termination (intercept update)
        self.nadir_point = None #initialize nadir-point as Empty
        self.term_gen = 1 #this is for nadir point termination
        self.extreme_points = None # initialize extremem points as Empty
        self.insights = [] # initialize as empty array to gather some information
        self.F_check = None
        self.feas_index = None
        self.infeas_index = None
        self.min_constr_value = []

        # for global termination
        self.termination_suggestion = [None,None] # save the generations at which termination is suggested
        self.termination_pop = [None,None] # save the population when the termination is suggested

        # initialize the termination archive as empty set.
        self.mu_D = []
        self.D_t = []
        self.S_t = []

        set_if_none(kwargs, 'pop_size', len(ref_dirs))
        set_if_none(kwargs, 'sampling', FloatRandomSampling())
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob=1.0, eta=20))
        set_if_none(kwargs, 'mutation', PolynomialMutation(prob=None, eta=20))
        set_if_none(kwargs, 'survival', None)
        set_if_none(kwargs, 'selection', RandomSelection())

        super().__init__(display=display, **kwargs)

        # initialized when problem is known
        self.ref_dirs = ref_dirs
    
    def _initialize(self):

        # nadir point estimation
        pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        self.evaluator.eval(self.problem, pop, algorithm=self)
        for i in range(200):
            off = self.mating.do(self.problem, pop, self.n_offsprings, algorithm=self)
            self.evaluator.eval(self.problem, off, algorithm=self)
            pop = Population.merge(pop, off)
            F = pop.get("F")
            fitness = np.zeros(F.shape)
            for i in range(len(F)):
                for j in range(len(F[i])):
                    fitness[i,j] = abs(F[i,j]) + 100*(sum(F[i]**2)-F[i,j]**2)
            survivors = []
            count = int(np.ceil(self.pop_size/self.problem.n_obj))
            for i in range(self.problem.n_obj):
                F = fitness[:,i]
                index = np.argsort(F)
                survivors.extend(index[:count])
            pop = pop[np.unique(survivors)]
        F = pop.get("F")
        fitness = np.zeros(F.shape)
        for i in range(len(F)):
            for j in range(len(F[i])):
                fitness[i,j] = abs(F[i,j]) + 100*(sum(F[i]**2)-F[i,j]**2)
        index = np.argmin(fitness, axis=0)
        extremes = np.array([ind.F for ind in pop[index]])
        nadir_point = []
        for i in range(self.problem.n_obj):
            nadir_point.append(extremes[i][i])
        ideal_point = np.min(pop.get("F"), axis=0)

        # initialize
        self.ideal_point = ideal_point
        self.nadir_point = nadir_point
        ref = np.zeros([self.pop_size, self.problem.n_obj])
        for i in range(self.pop_size):
            for j in range(self.problem.n_obj):
                ref[i,j] = self.ref_dirs[i,j]*(nadir_point[j]-ideal_point[j]) + ideal_point[j]
        self.ref = ref

        super()._initialize()

        # set population attributes (rank and crowding)
        for i in range(self.pop_size):
            self.pop[i].set("rank", None)
            self.pop[i].set("proximity", None)

    def _next(self):

        if self.pop[0].data["rank"] is None:
            for i in range(self.pop_size):
                self.pop[i].data["rank"] = rank(self, self.pop[i].F)
                self.pop[i].data["proximity"] = proximity(self, self.pop[i].F, self.pop[i].data["rank"])

        # do the mating using the current population
        self.mating.selection = TournamentSelection(func_comp=binary_tournament)
        self.off = self.mating.do(self.problem, self.pop, self.n_offsprings, algorithm=self)
        self.off.set("n_gen", self.n_gen)

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

        # assign rank and proximity
        for i in range(len(self.off)):
            self.off[i].set("rank", rank(self, self.off[i].F))
            self.off[i].set("proximity", proximity(self, self.off[i].F, self.off[i].data["rank"]))

        # merge the offsprings with the current population
        self.pop = Population.merge(self.pop, self.off)

        # the do survival selection
        self.pop = survival_selection(self)
        
def rank(self, F):
    for i in range(len(self.ref)):
        temp = sum(F<self.ref[i])
        if temp==self.problem.n_obj:
            return 1
        elif temp==0:
            return 3
    return 2

def proximity(self, F, rank):
    d = np.zeros(len(self.ref))
    for i in range(len(self.ref)):
        p = self.ref[i]
        if rank==1:
            d[i] = -1*((sum([(F[j]-p[j])**2 for j in range(self.problem.n_obj)]))**0.5)
        elif rank==2:
            d[i] = (sum([(max(F[j]-p[j],0))**2 for j in range(self.problem.n_obj)]))**0.5
        else:
            d[i] = (sum([(F[j]-p[j])**2 for j in range(self.problem.n_obj)]))**0.5
    return d

def binary_tournament(pop, P, algorithm, **kwargs):
    if P.shape[1] != 2:
        raise ValueError("Only implemented for binary tournament!")

    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):

        a, b = P[i, 0], P[i, 1]

        # if at least one solution is infeasible
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[i] = compare(a, pop[a].CV, b, pop[b].CV, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible
        else:
            S[i] = compare(a, pop[a].data["rank"], b, pop[b].data["rank"],
                            method='smaller_is_better')

            # if rank or domination relation didn't make a decision compare by crowding
            if np.isnan(S[i]):
                S[i] = compare(a, min(pop[a].get("proximity")), b, min(pop[b].get("proximity")),
                               method='smaller_is_better', return_random_if_equal=True)

    return S[:, None].astype(np.int, copy=False)

def survival_selection(self):

    survivors = []
    rank = 1
    F = [[],[],[]]
    F[0] = [i for i in range(len(self.pop)) if self.pop[i].data["rank"]==1]
    F[1] = [i for i in range(len(self.pop)) if self.pop[i].data["rank"]==2]
    F[2] = [i for i in range(len(self.pop)) if self.pop[i].data["rank"]==3]
    print("Ranks:",len(F[0]), len(F[1]), len(F[2]))
    while len(survivors) + len(F[rank-1]) < self.pop_size:
        survivors.extend(F[rank-1])
        rank += 1
    if len(survivors) + len(F[rank-1]) == self.pop_size:
        survivors.extend(F[rank-1])
    else:
        to_select = self.pop_size - len(survivors)
        W = self.ref
        Distance = cdist(W,W)
        Distance[np.eye(len(Distance))==1] = math.inf
        Delete = [False]*len(W)
        print(len(self.ref), to_select)
        for i in range(len(self.ref) - to_select):
            Remain = np.array([j for j in range(len(W)) if Delete[j]==False])
        #     print(Remain)
            Temp = np.sort(Distance, axis=1)
        #     print(Temp)
            for j in range(len(Remain)):
                check = Temp[:,j]
                if len(np.where(check==np.min(check))[0])==1:
                    index = j
                    break
                else:
                    index = 0
        #     print(Remain[index])
            Distance = np.delete(Distance, index, 0)
            Distance = np.delete(Distance, index, 1)
            Delete[Remain[index]] = True

        #selection of remaining solutions
        for i in range(self.pop_size):
            if Delete[i]==False:
                left_solutions = np.array([i for i in range(len(self.pop)) if i not in survivors])
                left_pop = self.pop[left_solutions]
                dist = left_pop.get("proximity")
                dist = dist[:,i]
                index = np.argmin(dist)
                survivors.append(left_solutions[index])

    print(len(survivors))
    return self.pop[survivors]

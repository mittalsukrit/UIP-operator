import numpy as np
from scipy.spatial.distance import cdist
import warnings
from numpy.linalg import LinAlgError
from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.factory import get_decomposition
from pymoo.model.individual import Individual
from pymoo.model.population import Population
from pymoo.operators.selection.random_selection import RandomSelection
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.misc import set_if_none
from pymoo.util.normalization import normalize
from pymoo.util.function_loader import load_function
import math
import hvwfg


# =========================================================================================================
# Implementation
# =========================================================================================================

class MDMOEA(GeneticAlgorithm):

    def __init__(self,
                 display=MultiObjectiveDisplay(),
                 **kwargs):
        """

        HFIDEA Algorithm.

        Parameters
        ----------
        ref_dirs
        display
        kwargs
        """

        set_if_none(kwargs, 'pop_size', 100)
        set_if_none(kwargs, 'sampling', FloatRandomSampling())
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob=1.0, eta=20))
        set_if_none(kwargs, 'mutation', PolynomialMutation(prob=None, eta=20))
        set_if_none(kwargs, 'survival', None)
        set_if_none(kwargs, 'selection', RandomSelection())

        super().__init__(display=display, **kwargs)

    def _initialize(self):

        super()._initialize()
        
        # initialize the analog of temperature T>0
        self.T = 10000

        self.ideal_point, self.nadir_point = None, None

        for i in range(self.pop_size):
            self.pop[i].set("LR", None)
            self.pop[i].set("LS", None)
            self.pop[i].set("d", None)
            self.pop[i].set("fitness", None)

    def _next(self):

        # calculate the L-rank values of all solutions
        # F = self.pop.get("F")
        # print('Hypervolume: ', hvwfg.wfg(F, np.array([self.pop_size/(self.pop_size-1)]*self.problem.n_obj)))
        # for i in range(self.pop_size):
            # print(self.pop[i].data["LR"], self.pop[i].data["LS"], self.pop[i].data["d"])

        # evaluate the fitness of all solutions
        worst = calculate_fitness(self)
        # if self.pop[0].data["LR"] is None:
        #     worst = calculate_fitness(self)
        # else:
        temp = np.array([ind.data["fitness"] for ind in self.pop])
        print([np.min(temp),np.mean(temp),np.max(temp)])
        #     worst = np.argmax(temp)

        # do the mating using the current population
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

        # select new individuals
        for new in self.off:
            count_L_dom = 0
            for j in range(self.pop_size):
                n_j = sum(new.F>self.pop[j].F)
                n_i = sum(new.F<self.pop[j].F)
                if n_j >= n_i:
                    f_new = (new.F - self.ideal_point)/(self.nadir_point - self.ideal_point)
                    f_j = (self.pop[j].F - self.ideal_point)/(self.nadir_point - self.ideal_point)
                    if np.linalg.norm(f_j) < np.linalg.norm(f_new):
                        count_L_dom += 1
                # if n_j > 0 and n_i == 0:
                #     count_L_dom += 1
            LR_new = count_L_dom
            if LR_new < self.pop[worst].data["LR"]:
                self.pop[worst].X = new.X
                self.pop[worst].F = new.F
                # worst = calculate_fitness(self)
                worst = update_fitness(self, worst)
            else:
                d_new = crowding(new.F, LR_new, self.pop.get("F"), self.pop.get("LR"), self.problem.n_obj)
                if LR_new == self.pop[worst].data["LR"] and d_new > self.pop[worst].data["d"]:
                    self.pop[worst].X = new.X
                    self.pop[worst].F = new.F
                    # worst = calculate_fitness(self)
                    worst = update_fitness(self, worst)
                elif math.exp((self.pop[worst].data["LR"] - LR_new)/self.T) > np.random.random():
                    self.pop[worst].X = new.X
                    self.pop[worst].F = new.F
                    # worst = calculate_fitness(self)
                    worst = update_fitness(self, worst)
        
        # update rank of all individuals
        # F = self.pop.get("F")
        # F_min, F_max = np.min(F, axis=0), np.max(F, axis=0) # ideal-point and nadir-point
        # self.ideal_point, self.nadir_point = F_min, F_max
        # for i in range(self.pop_size):
        #     count_L_dom = 0
        #     for j in range(self.pop_size): # to check if (j) L-dominates (i)
        #         if j!=i:
        #             n_j = sum(F[i]>F[j]) # j is better than i
        #             n_i = sum(F[i]<F[j]) # i is better than j
        #             if n_j > n_i:
        #                 f_i = (F[i] - F_min)/(F_max - F_min)
        #                 f_j = (F[j] - F_min)/(F_max - F_min)
        #                 if np.linalg.norm(f_j) < np.linalg.norm(f_i):
        #                     count_L_dom += 1
        #             # if n_j > 0 and n_i == 0:
        #             #     count_L_dom += 1
        #     self.pop[i].data["LR"] = count_L_dom
        #     self.pop[i].data["fitness"] = self.pop[i].data["LR"] - self.T*self.pop[i].data["LS"] - self.pop[i].data["d"]


def calculate_fitness(self):

    F = self.pop.get("F")
    F_min, F_max = np.min(F, axis=0), np.max(F, axis=0) # ideal-point and nadir-point
    self.ideal_point, self.nadir_point = F_min, F_max
    for i in range(self.pop_size):
        count_L_dom = 0
        for j in range(self.pop_size): # to check if (j) L-dominates (i)
            if j!=i:
                n_j = sum(F[i]>F[j]) # j is better than i
                n_i = sum(F[i]<F[j]) # i is better than j
                if n_j > n_i:
                    f_i = (F[i] - F_min)/(F_max - F_min)
                    f_j = (F[j] - F_min)/(F_max - F_min)
                    if np.linalg.norm(f_j) < np.linalg.norm(f_i):
                        count_L_dom += 1
                # if n_j > 0 and n_i == 0:
                #     count_L_dom += 1
        self.pop[i].data["LR"] = count_L_dom
    ranks = np.array([ind.data["LR"] for ind in self.pop])
    
    # evaluate the fitness of each individual
    Z = 0
    for i in range(self.pop_size):
        Z += math.exp(-1*self.pop[i].data["LR"]/self.T)
    
    for i in range(self.pop_size):
        self.pop[i].data["d"] = crowding(F[i], self.pop[i].data["LR"], F, ranks, self.problem.n_obj)
        pt = math.exp(-1*self.pop[i].data["LR"]/self.T)/Z
        S = -1*pt*math.log(pt)
        self.pop[i].data["LS"] = S
        # fitness = self.pop[i].data["LR"] - 1*self.pop[i].data["LS"] - self.pop[i].data["d"]
        fitness = self.pop[i].data["LR"] - self.T*self.pop[i].data["LS"] - self.pop[i].data["d"]
        self.pop[i].data["fitness"] = fitness

    fitness = [ind.data["fitness"] for ind in self.pop]
    worst = np.argmax(fitness)
    return worst

def update_fitness(self, worst):

    F = self.pop.get("F")
    F_min, F_max = np.min(F, axis=0), np.max(F, axis=0) # ideal-point and nadir-point
    self.ideal_point, self.nadir_point = F_min, F_max
    for i in [worst]:
        count_L_dom = 0
        for j in range(self.pop_size): # to check if (j) L-dominates (i)
            if j!=i:
                n_j = sum(F[i]>F[j]) # j is better than i
                n_i = sum(F[i]<F[j]) # i is better than j
                if n_j > n_i:
                    f_i = (F[i] - F_min)/(F_max - F_min)
                    f_j = (F[j] - F_min)/(F_max - F_min)
                    if np.linalg.norm(f_j) < np.linalg.norm(f_i):
                        count_L_dom += 1
                # if n_j > 0 and n_i == 0:
                #     count_L_dom += 1
        self.pop[i].data["LR"] = count_L_dom
    ranks = np.array([ind.data["LR"] for ind in self.pop])
    
    # evaluate the fitness of each individual
    Z = 0
    for i in range(self.pop_size):
        Z += math.exp(-1*self.pop[i].data["LR"]/self.T)
    
    for i in [worst]:
        self.pop[i].data["d"] = crowding(F[i], self.pop[i].data["LR"], F, ranks, self.problem.n_obj)
        pt = math.exp(-1*self.pop[i].data["LR"]/self.T)/Z
        S = -1*pt*math.log(pt)
        self.pop[i].data["LS"] = S
        # fitness = self.pop[i].data["LR"] - 1*self.pop[i].data["LS"] - self.pop[i].data["d"]
        fitness = self.pop[i].data["LR"] - self.T*self.pop[i].data["LS"] - self.pop[i].data["d"]
        self.pop[i].data["fitness"] = fitness

    fitness = [ind.data["fitness"] for ind in self.pop]
    worst = np.argmax(fitness)
    return worst
        
def crowding(f, rank, F, ranks, M):
    F_rank = np.array([F[i] for i in range(len(F)) if ranks[i]==rank])
    # F_rank = F # computing crowding distance considering all solutions at the same time
    if len(F_rank)>=2:
        F_min, F_max = np.min(F_rank, axis=0), np.max(F_rank, axis=0)
        d = 0
        for i in range(M):
            fr = F_rank[:,i]
            if f[i]<=F_min[i] or f[i]>=F_max[i]:
                return 1000000 #math.inf
            else:
                d += (min(fr[fr>f[i]]) - max(fr[fr<f[i]]))/(F_max[i] - F_min[i])
        return d
    else:
        return 1000000 #math.inf
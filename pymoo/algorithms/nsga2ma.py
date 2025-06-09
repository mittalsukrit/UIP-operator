import numpy as np

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.docs import parse_doc_string
from pymoo.model.individual import Individual
from pymoo.model.survival import Survival
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.operators.selection.tournament_selection import compare, TournamentSelection
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.dominator import Dominator
from pymoo.util.misc import find_duplicates, has_feasible
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.factory import get_reference_directions 
from scipy.spatial.distance import cdist,euclidean
from pymoo.util.normalization import normalize



# ---------------------------------------------------------------------------------------------------------
# Binary Tournament Selection Function
# ---------------------------------------------------------------------------------------------------------


def binary_tournament(pop, P, algorithm, **kwargs):
    if P.shape[1] != 2:
        raise ValueError("Only implemented for binary tournament!")

    tournament_type = algorithm.tournament_type
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):

        a, b = P[i, 0], P[i, 1]

        # if at least one solution is infeasible
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[i] = compare(a, pop[a].CV, b, pop[b].CV, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible
        else:

            if tournament_type == 'comp_by_dom_and_crowding':
                rel = Dominator.get_relation(pop[a].F, pop[b].F)
                if rel == 1:
                    S[i] = a
                elif rel == -1:
                    S[i] = b

            elif tournament_type == 'comp_by_rank_and_crowding':
                S[i] = compare(a, pop[a].data["rank"], b, pop[b].data["rank"],method='smaller_is_better')
                # S[i] = compare(a, pop[a].rank, b, pop[b].rank,method='smaller_is_better')

            else:
                raise Exception("Unknown tournament type.")

            # if rank or domination relation didn't make a decision compare by crowding
            if np.isnan(S[i]):
                S[i] = compare(a, pop[a].get("crowding"), b, pop[b].get("crowding"),
                               method='larger_is_better', return_random_if_equal=True)

    return S[:, None].astype(np.int, copy=False)


# =========================================================================================================
# Implementation
# =========================================================================================================


class NSGA2MA(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 cone_dom_alpha=0,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=binary_tournament),
                 crossover=SimulatedBinaryCrossover(eta=15, prob=0.9),
                 mutation=PolynomialMutation(prob=None, eta=20),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 display=MultiObjectiveDisplay(),
                 **kwargs):
        """

        Parameters
        ----------
        pop_size : {pop_size}
        sampling : {sampling}
        selection : {selection}
        crossover : {crossover}
        mutation : {mutation}
        eliminate_duplicates : {eliminate_duplicates}
        n_offsprings : {n_offsprings}

        """

        kwargs['individual'] = Individual(rank=np.inf, crowding=-1)
        super().__init__(pop_size=pop_size,
                         cone_dom_alpha=cone_dom_alpha,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=RankAndCrowdingSurvival(),
                         eliminate_duplicates=eliminate_duplicates,
                         n_offsprings=n_offsprings,
                         display=display,
                         **kwargs)

        NSGA2MA.cone_dom_alpha = cone_dom_alpha
        print(NSGA2MA.cone_dom_alpha)
        self.tournament_type = 'comp_by_rank_and_crowding'
        self.ref_dirs = get_reference_directions("das-dennis",3,n_partitions=23) #3-objective for population size of 300

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.pop[self.pop.get("rank") == 0]


# ---------------------------------------------------------------------------------------------------------
# Survival Selection
# ---------------------------------------------------------------------------------------------------------


class RankAndCrowdingSurvival(Survival):

    def __init__(self) -> None:
        super().__init__(filter_infeasible=True)

    def _do(self, problem, pop, n_survive, cone_dom_alpha, D=None, **kwargs):

        # get the objective space values and objects
        F = pop.get("F").astype(np.float, copy=False)
        F = normalize(F)

        #change F to F_CD (Cone-Domination)        
        a = cone_dom_alpha
        print(a)
        M = problem.n_obj
        dom_mat = (1-a)*np.eye(M) + (a/M)*np.ones([M,M])
        F = np.array([np.matmul(dom_mat,f[:]) for f in F])

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = NonDominatedSorting().do(F, n_stop_if_ranked=n_survive)

        #get back the original F-values
        F = pop.get("F").astype(np.float, copy=False)
        F = normalize(F)
        # to rule out convergence from the diversity measure.
        F = np.array([ind/sum(ind) for ind in F])

        for k, front in enumerate(fronts):

            # calculate the crowding distance of the front
            crowding_of_front = np.zeros(len(front))
            # crowding_of_front = calc_crowding_distance(F[front, :])

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > n_survive:
                # I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                # I = I[:(n_survive - len(survivors))]

                #Minimum distance selection formula - add extreme points
                # extremes = np.argmin(F[front, :],axis=0)
                extremes = get_extreme_points_c(F[front, :])
                extremes = [row for row in extremes]
                if len(survivors) + len(extremes) >= n_survive:
                    leftover = n_survive - len(survivors)
                    survivors.extend(front[np.random.permutation(extremes)[:leftover]])
                else:
                    survivors.extend(front[extremes])
                    leftover = n_survive - len(survivors)
                    for iterating in range(leftover):
                        ref = F[front[extremes], :]
                        audience = np.array([au for au in range(len(front)) if au not in extremes])
                        F_audience = F[front[audience], :]
                        dist = [min(cdist([f],ref)[0]) for f in F_audience]
                        # dist = [sum(np.sort(cdist([f],ref)[0])[:M]) for f in F_audience]
                        index = np.argmax(dist)
                        # print(len(front),len(audience),len(extremes),index)
                        survivors.append(front[audience[index]])
                        extremes.append(audience[index])
                        if len(survivors) >= n_survive:
                            break


            # otherwise take the whole front unsorted
            else:
                I = np.arange(len(front))
                survivors.extend(front[I])

            # extend the survivors by all or selected individuals
            # survivors.extend(front[I])

        for index in survivors:
            base_rank = pop[index].data["rank"]
            f = pop[index].F
            other_f = np.array([pop[i].F for i in range(len(survivors)) if i!=index and pop[i].data["rank"]==base_rank])
            dist = np.array([euclidean(f,others) for others in other_f])
            if len(dist)>0:
                dist = np.sort(dist)[min(len(dist)-1,M-1)] #k-th neighbour distance, where k = M-1
                pop[index].data["crowding"] = dist
            else:
                pop[index].data["crowding"] = 0 #Doesn't matter since there is only one solutions in this rank

        return pop[survivors]

def get_alpha():
    global self
    return NSGA2MA.cone_dom_alpha

def get_extreme_points_c(F):

    ideal_point = np.min(F,axis=0)

    # calculate the asf which is used for the extreme point decomposition
    weights = np.eye(F.shape[1])
    weights[weights == 0] = 1e6

    _F = F

    # use __F because we substitute small values to be 0
    __F = _F - ideal_point
    __F[__F < 1e-3] = 0

    # update the extreme points for the normalization having the highest asf value each
    F_asf = np.max(__F * weights[:, None, :], axis=2)

    I = np.argmin(F_asf, axis=1)
    extreme_points = _F[I, :]

    I = np.unique(I)

    return I


parse_doc_string(NSGA2MA.__init__)

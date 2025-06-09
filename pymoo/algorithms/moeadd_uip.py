import numpy as np
from scipy.spatial.distance import cdist
import math
import warnings
from numpy.linalg import LinAlgError
from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.factory import get_decomposition
from pymoo.model.individual import Individual
from pymoo.model.population import Population,pop_from_array_or_individual
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.misc import set_if_none
from pymoo.model.survival import Survival
from pymoo.util.normalization import normalize, denormalize
from copy import deepcopy
from pymoo.util.function_loader import load_function
from pymoo.util.misc import intersect, has_feasible
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.factory import get_decomposition


# =========================================================================================================
# Implementation
# =========================================================================================================

class MOEADD(GeneticAlgorithm):

    def __init__(self,
                 ref_dirs,
                 n_neighbors=20,
                 decomposition='auto',
                 prob_neighbor_mating=0.7,
                 display=MultiObjectiveDisplay(),
                 **kwargs):
        """

        MOEA/DD Algorithm.

        Parameters
        ----------
        ref_dirs
        display
        kwargs
        """

        self.n_neighbors = n_neighbors
        self.prob_neighbor_mating = prob_neighbor_mating
        

        set_if_none(kwargs, 'pop_size', len(ref_dirs))
        set_if_none(kwargs, 'sampling', FloatRandomSampling())
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob=1.0, eta=20))
        set_if_none(kwargs, 'mutation', PolynomialMutation(prob=None, eta=20))
        set_if_none(kwargs, 'survival', None)
        set_if_none(kwargs, 'selection', None)

        del kwargs['survival']
        survival = MOEADDSurvival(ref_dirs)

        super().__init__(display=display, survival=survival, **kwargs)

        # initialized when problem is known
        self.ref_dirs = ref_dirs

    def _initialize(self):

        super()._initialize()

        self.all_F = []
        self.all_CV = []

        self.ideal_point = np.min(self.pop.get("F"), axis=0)
        self.nadir_point = np.max(self.pop.get("F"), axis=0)

        # Assign each weight with a randomly selected solution
        self.assoc_index = np.random.permutation(self.pop_size)

        # Find the 'T' neighboring weights
        T = self.n_neighbors
        matrix = [[calc_angle(a,b) for a in self.ref_dirs] for b in self.ref_dirs]
        self.neighbor_weights = [np.argsort(row)[1:T+1] for row in matrix]

    def _next(self):
        repair, crossover, mutation = self.repair, self.mating.crossover, self.mating.mutation

        print(self.co_frequency, self.do_frequency)

        """Check if termination has been suggested. If yes, terminate"""
        check = [val for val in self.termination_suggestion if val is None]
        if len(check)==0 and self.is_co_learn and self.is_do_learn:
            self.termination.force_termination = True
            return

        """Should we start CO?"""
        if self.is_co_learn and self.co_start:
            GeneticAlgorithm.update_co_target(self)
            if self.n_gen>self.collection+1 and self.n_gen-self.last_co_repair>=self.co_frequency:
                print('CO learning now')
                lower_bound,upper_bound, RF_model = GeneticAlgorithm.co_learn(self)

        """Set the trigger of non-domination"""
        if self.do_trigger1==False and self.is_do_learn:
            if self.problem.n_constr==0:
                self.do_trigger1 = True
            else:
                CV = self.pop.get("CV")
                if max(CV)==0:
                    self.do_trigger1 = True

        """Store the parent population in Q"""
        if self.is_do_learn and self.do_trigger1:
            Q_term = self.pop.get("F")
            parent_niches = self.pop.get("niche")
            parent_niches = np.unique(parent_niches)

        """Call the Learn function"""
        if self.is_do_learn and self.do_trigger2 and self.n_gen-self.last_do_learn>=self.do_frequency:
            print('DO learning now')
            GeneticAlgorithm.do_learn(self)

        #set a temporary solution set
        Sc = []
        CV = [ind.CV[0] for ind in self.pop]

        for i in range(self.pop_size):

            if np.random.random() < self.prob_neighbor_mating and max(CV)==0:
                # find the neighboring solutions 
                Q = [j for j in range(self.pop_size) if self.pop[j].data["niche"] in self.neighbor_weights[i]]
                if len(Q) < 2:
                    Q = [j for j in range(self.pop_size)]
            else:
                Q = [j for j in range(self.pop_size)]
            
            # generate a new soution
            parents = np.random.permutation(Q)[:crossover.n_parents]
            # parents = np.array([100000]*crossover.n_parents) #set some arbitrary value
            # while len(np.unique(parents))<crossover.n_parents:
            #     parents = np.random.choice(Q,crossover.n_parents)
            off = crossover.do(self.problem, self.pop, parents[None, :])
            off = mutation.do(self.problem, off)
            # Sc.append(np.random.choice(off))
            Sc.append(off[0])

        # evaluate Sc and merge the population 
        off_x = np.array([ind.X for ind in Sc])
        offs = Population(n_individuals=self.pop_size)
        offs = offs.new("X", off_x)

        """repair step:"""
        if self.n_gen>self.collection+1 and self.is_co_learn and self.n_gen-self.last_co_repair>=self.co_frequency and self.co_start:
            print('CO repairing now')
            self.last_co_repair = self.n_gen
            offs = MOEADD.co_repair(self, lower_bound,upper_bound, RF_model, offs)

        """Call the Repair function and replace offspring"""
        if self.is_do_learn and self.do_trigger2 and self.n_gen-self.last_do_learn>=self.do_frequency:
            temp = [1 for model in self.do_models if model is None]
            if sum(temp)==0:
                print('DO repairing now')
                self.last_do_learn = self.n_gen
                new_solutions = GeneticAlgorithm.do_repair(self)
                sequence = np.arange(int(self.pop_size/2)) #DEXTER - offspring generated are already randomized - doesn't make sense to use another random input there
                for i in range(len(new_solutions)):
                    offs[sequence[i]].X = np.array(new_solutions[i])

        self.evaluator.eval(self.problem, offs, algorithm=self)
        if self.is_co_learn:
            self.child.append(offs)
        self.pop = Population.merge(self.pop, offs)

        """The do survival selection"""
        self.pop = self.survival.do(self.problem, self.pop, self.pop_size, algorithm=self,
                                        n_min_infeas_survive=self.min_infeas_pop_size)

        """Update termination archives"""
        if self.do_trigger1==True:
            GeneticAlgorithm.update_termination_archives(self, Q_term)

        """co_frequency adaptation"""
        df1 = self.pop.get("F")
        df2 = np.array([ind.F for ind in offs])
        df2 = df2[int(self.pop_size/2):]
        count = 0
        for val in df1:
            if sum(sum(df2==val))>0:
                count+=1
        self.co_survived.append(count)
        if self.n_gen>self.collection+1 and self.is_co_learn and self.n_gen-self.last_co_repair==0 and self.co_start:
            if self.co_survived[-1]>self.co_survived[-2] and self.co_frequency>2:
                self.co_frequency-=1
            elif self.co_survived[-1]<self.co_survived[-2]:
                self.co_frequency+=1

        """do_frequency adaptation"""
        df1 = self.pop.get("F")
        df2 = np.array([ind.F for ind in offs])
        df2 = df2[:int(self.pop_size/2)]
        count = 0
        for val in df1:
            if sum(sum(df2==val))>0:
                count+=1
        self.do_survived.append(count)
        if self.is_do_learn and self.do_trigger2 and self.n_gen-self.last_do_learn==0:
            temp = [1 for model in self.do_models if model is None]
            if sum(temp)==0:
                if self.do_survived[-1]>self.do_survived[-2] and self.do_frequency>2:
                    self.do_frequency-=1
                elif self.do_survived[-1]<self.do_survived[-2]:
                    self.do_frequency+=1
        
        """Check for starting DL operator"""
        if self.is_do_learn and self.do_trigger1==True and self.co_start==False:
            ranks = [ind.data["rank"] for ind in self.pop]
            # print(ranks)
            if max(ranks)==0:
                self.co_start = True
            # if termination_flag and max(ranks)==0:
        if self.is_do_learn and self.do_trigger1==True:
            if self.do_trigger2==False:
                termination_flag = GeneticAlgorithm.check_for_termination(self, 2, 20)
                if termination_flag:
                    self.do_trigger2 = True

        """Check for termination"""
        if self.termination_suggestion[0] is None:
            term_cond = GeneticAlgorithm.check_for_termination(self, 2, 20) # termination-2 parameters
            if term_cond:
                self.termination_suggestion[0] = self.n_gen
                print("mild stabilization has reached at gen: "+str(self.n_gen))

        if self.termination_suggestion[1] is None:
            term_cond = GeneticAlgorithm.check_for_termination(self, 3, 20) # termination-2 parameters
            if term_cond:
                self.termination_suggestion[1] = self.n_gen
                print("intermediate stabilization has reached at gen: "+str(self.n_gen))
        
        if self.termination_suggestion[2] is None:
            term_cond = GeneticAlgorithm.check_for_termination(self, 3, 50) # termination-2 parameters
            if term_cond:
                self.termination_suggestion[2] = self.n_gen
                print("strict stabilization has reached at gen: "+str(self.n_gen))

    def co_repair(self, lower_bound, upper_bound, regr, offs):
        child = np.array([offs[i].X for i in range (self.pop_size)])
        child = normalize(child,x_min=lower_bound,x_max=upper_bound)
        child = regr.predict(child)
        child = denormalize(child,lower_bound,upper_bound)
        for i in range (0,int(self.pop_size*self.co_repair_fraction)):
            # index = np.random.randint(0,self.pop_size-1)
            index = i + int(self.pop_size/2) #DEXTER - since the mating is random, doesn't make sense to involve another random selection process
            self.co_boost = 1 + 0.5*np.random.random()
            modification = np.array(self.co_boost*(np.array(child[index])-offs[index].X))
            original_child = deepcopy(offs[index].X)
            offs[index].X += modification

            """Near Bound Restoration"""
            for j in range (self.problem.n_var):
                lower_dist = abs(original_child[j]-self.problem.xl[j])
                upper_dist = abs(original_child[j]-self.problem.xu[j])
                if lower_dist<upper_dist:
                    dist=lower_dist
                else:
                    dist=upper_dist
                length = abs(self.problem.xu[j]-self.problem.xl[j])
                if dist<0.01*length:
                    offs[index].X[j] -= 1.0*(original_child[j])
                
            """Variable Boundary Repair"""
            offs[index].X = GeneticAlgorithm.co_boundary_repair(original_child, offs[index].X,self.problem.xl,self.problem.xu,0)
            # offs[index].X = GeneticAlgorithm.boundary_repair(original_child, offs[index].X, self.problem)
        return offs

class MOEADDSurvival(Survival):

    def __init__(self, ref_dirs):
        super().__init__(filter_infeasible=True)
        self.ref_dirs = ref_dirs
        self.extreme_points = None
        self.intercepts = None
        self.nadir_point = None
        self.opt = None
        self.ideal_point = np.full(ref_dirs.shape[1], np.inf)
        self.worst_point = np.full(ref_dirs.shape[1], -np.inf)

    def _do(self, problem, pop, n_survive, D=None, **kwargs):
        # attributes to be set after the survival
        F = pop.get("F")

        # Update the ideal point
        self.ideal_point = np.min(np.vstack((self.ideal_point, F)), axis=0)
        self.worst_point = np.max(np.vstack((self.worst_point, F)), axis=0)

        # get Pareto non domination levels
        fronts, rank = NonDominatedSorting().do(F, return_rank=True, n_stop_if_ranked=n_survive)
        non_dominated, last_front = fronts[0], fronts[-1]
        print("ND:", len(non_dominated))

        # find the extreme points for normalization
        self.extreme_points = get_extreme_points_c(F[non_dominated, :], self.ideal_point,
                                                   extreme_points=self.extreme_points)

        # find the intercepts for normalization and do backup if gaussian elimination fails
        worst_of_population = np.max(F, axis=0)
        worst_of_front = np.max(F[non_dominated, :], axis=0)

        self.nadir_point = get_nadir_point(self.extreme_points, self.ideal_point, self.worst_point,
                                           worst_of_population, worst_of_front)

        #  consider only the population until we come to the splitting front
        I = np.concatenate(fronts)
        pop, rank, F = pop[I], rank[I], F[I]

        # update the front indices for the current population
        counter = 0
        for i in range(len(fronts)):
            for j in range(len(fronts[i])):
                fronts[i][j] = counter
                counter += 1
        last_front = fronts[-1]

        # clustering
        niche_of_individuals, dist_to_niche, dist_matrix = \
            associate_to_niches(F, self.ref_dirs, self.ideal_point, self.nadir_point)

        F_norm = normalize(F, x_min = self.ideal_point, x_max = self.nadir_point)
        # decomposition = get_decomposition("pbi", theta=5)

        # pbi 
        # pbi = [decomposition.do(F_norm[i], weights=self.ref_dirs[niche_of_individuals[i]], ideal_point=np.array([0]*F.shape[-1]))[0,0] for i in range(len(F))]
        pbi = []
        for i in range(len(F)):
            decomposition = get_decomposition("pbi", theta=5)
            temp = decomposition.do(F_norm[i], weights=self.ref_dirs[niche_of_individuals[i]], ideal_point=np.array([0]*F.shape[-1]))[0,0]
            pbi.append(temp)

        # attributes of a population
        pop.set('rank', rank,
                'niche', niche_of_individuals,
                'dist_to_niche', dist_to_niche,#)
                'pbi', pbi,
                'ideal_point', np.array([self.ideal_point]*len(pop)), #DEXTER
                'nadir_point', np.array([self.nadir_point]*len(pop))) #DEXTER

        # set the optimum, first front and closest to all reference directions
        closest = np.unique(dist_matrix[:, np.unique(niche_of_individuals)].argmin(axis=0))
        self.opt = pop[intersect(fronts[0], closest)]

        # if we need to select individuals to survive
        if len(pop) > n_survive:
            clusters = []
            clusters_with_last_rank = []
            for i in range(len(self.ref_dirs)):
                temp = []
                flag = False
                for j in range(len(pop)):
                    if pop[j].data['niche'] == i:
                        temp.append(j)
                        if pop[j].data['rank'] == max(rank):
                            flag = True
                clusters_with_last_rank.append(i)
                clusters.append(temp)

            to_delete = len(pop) - n_survive
            selected_clusters = [clusters[i] for i in range(len(clusters)) if i in clusters_with_last_rank]
            length_clusters = [len(clusters[i]) for i in range(len(clusters)) if i in clusters_with_last_rank]
            delete_indices = []
            for i in range(to_delete):
                cluster_index = np.argmax(length_clusters)
                delete_indices.append(selected_clusters[cluster_index][-1])
                selected_clusters[cluster_index] = selected_clusters[cluster_index][:-1]
                length_clusters[cluster_index] -= 1
            survivors = [i for i in range(len(pop)) if i not in delete_indices]

            pop = pop[survivors]

        return pop


def get_extreme_points_c(F, ideal_point, extreme_points=None):
    # calculate the asf which is used for the extreme point decomposition
    weights = np.eye(F.shape[1])
    weights[weights == 0] = 1e6

    # add the old extreme points to never loose them for normalization
    _F = F
    if extreme_points is not None:
        _F = np.concatenate([extreme_points, _F], axis=0)

    # use __F because we substitute small values to be 0
    __F = _F - ideal_point
    __F[__F < 1e-3] = 0

    # update the extreme points for the normalization having the highest asf value each
    F_asf = np.max(__F * weights[:, None, :], axis=2)

    I = np.argmin(F_asf, axis=1)
    extreme_points = _F[I, :]

    return extreme_points


def get_nadir_point(extreme_points, ideal_point, worst_point, worst_of_front, worst_of_population):
    try:

        # find the intercepts using gaussian elimination
        M = extreme_points - ideal_point
        b = np.ones(extreme_points.shape[1])
        plane = np.linalg.solve(M, b)

        warnings.simplefilter("ignore")
        intercepts = 1 / plane

        nadir_point = ideal_point + intercepts

        # check if the hyperplane makes sense
        if not np.allclose(np.dot(M, plane), b) or np.any(intercepts <= 1e-6):
            raise LinAlgError()

        # if the nadir point should be larger than any value discovered so far set it to that value
        # NOTE: different to the proposed version in the paper
        b = nadir_point > worst_point
        nadir_point[b] = worst_point[b]

    except LinAlgError:

        # fall back to worst of front otherwise
        nadir_point = worst_of_front

    # if the range is too small set it to worst of population
    b = nadir_point - ideal_point <= 1e-6
    nadir_point[b] = worst_of_population[b]

    return nadir_point


def niching(pop, n_remaining, niche_count, niche_of_individuals, dist_to_niche):
    survivors = []

    # boolean array of elements that are considered for each iteration
    mask = np.full(len(pop), True)

    while len(survivors) < n_remaining:

        # number of individuals to select in this iteration
        n_select = n_remaining - len(survivors)

        # all niches where new individuals can be assigned to and the corresponding niche count
        next_niches_list = np.unique(niche_of_individuals[mask])
        next_niche_count = niche_count[next_niches_list]

        # the minimum niche count
        min_niche_count = next_niche_count.min()

        # all niches with the minimum niche count (truncate if randomly if more niches than remaining individuals)
        next_niches = next_niches_list[np.where(next_niche_count == min_niche_count)[0]]
        next_niches = next_niches[np.random.permutation(len(next_niches))[:n_select]]

        for next_niche in next_niches:

            # indices of individuals that are considered and assign to next_niche
            next_ind = np.where(np.logical_and(niche_of_individuals == next_niche, mask))[0]

            # shuffle to break random tie (equal perp. dist) or select randomly
            np.random.shuffle(next_ind)

            if niche_count[next_niche] == 0:
                next_ind = next_ind[np.argmin(dist_to_niche[next_ind])]
            else:
                # already randomized through shuffling
                next_ind = next_ind[0]

            # add the selected individual to the survivors
            mask[next_ind] = False
            survivors.append(int(next_ind))

            # increase the corresponding niche count
            niche_count[next_niche] += 1

    return survivors


def associate_to_niches(F, niches, ideal_point, nadir_point, utopian_epsilon=0.0):
    utopian_point = ideal_point - utopian_epsilon

    denom = nadir_point - utopian_point
    denom[denom == 0] = 1e-12

    # normalize by ideal point and intercepts
    N = (F - utopian_point) / denom
    dist_matrix = load_function("calc_perpendicular_distance")(N, niches)

    niche_of_individuals = np.argmin(dist_matrix, axis=1)
    dist_to_niche = dist_matrix[np.arange(F.shape[0]), niche_of_individuals]

    return niche_of_individuals, dist_to_niche, dist_matrix


def calc_niche_count(n_niches, niche_of_individuals):
    niche_count = np.zeros(n_niches, dtype=np.int)
    index, count = np.unique(niche_of_individuals, return_counts=True)
    niche_count[index] = count
    return niche_count

        

def calc_angle(a,b):
    # print(a,b)
    val = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
    # print(val)
    if val>1: # to remove computation errors. 
        val = 1
    elif val<0:
        val = 0
    return math.acos(val)
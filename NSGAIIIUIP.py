# imports
import numpy as np
import warnings
from numpy.linalg import LinAlgError
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.population import Population
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection, compare
from pymoo.util.normalization import normalize
from pymoo.util.function_loader import load_function
from pymoo.core.survival import Survival
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.misc import intersect
from copy import deepcopy
import math
from scipy.spatial.distance import cdist, euclidean
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from pymoo.util.normalization import normalize, denormalize
from pymoo.util.misc import calc_perpendicular_distance

# =========================================================================================================
# Implementation
# =========================================================================================================

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

    return S[:, None].astype(int)

class NSGA3(GeneticAlgorithm):

    def __init__(self,
                 ref_dirs,
                 pop_size=None,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=comp_by_cv_then_random),
                 crossover=SBX(eta=30, prob=1.0),
                 mutation=PM(eta=20),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 output=MultiObjectiveOutput(),
                 is_co_learn = False,
                 is_do_learn = False,
                 co_repair_fraction = 0.5,
                 do_repair_fraction = [0.25, 0.25],
                 **kwargs):
        """

        NSGA-III-UIP Algorithm.

        Parameters
        ----------

        ref_dirs : {ref_dirs}
        pop_size : int (default = None)
            By default the population size is set to None which means that it will be equal to the number of reference
            line. However, if desired this can be overwritten by providing a positive number.
        sampling : {sampling}
        selection : {selection}
        crossover : {crossover}
        mutation : {mutation}
        eliminate_duplicates : {eliminate_duplicates}
        n_offsprings : {n_offsprings}
        """

        self.ref_dirs = ref_dirs

        """For overall termination"""
        self.termination_suggestion = [None, None]
        self.termination_pop = None

        """Archives for stabilization tracking algorithm"""
        self.mu_D = []
        self.D_t = []
        self.S_t = []
        self.Q = None

        """Set the CO related parameters"""
        self.collection = 5
        self.co_target = None
        self.co_start = False
        self.co_survived = []
        self.last_co_repair = 0
        self.is_co_learn = is_co_learn
        self.co_repair_fraction = co_repair_fraction
        self.history = []

        """Set the DO related parameters"""
        self.do_state = [0, 0, 0]
        self.do_models = None
        self.do_survived = []
        self.last_do_learn = 0
        self.is_do_learn = is_do_learn
        self.do_repair_fraction = do_repair_fraction

        if pop_size is None:
            pop_size = len(self.ref_dirs)

        elif pop_size < len(self.ref_dirs):
            print(
                f"WARNING: pop_size={pop_size} is less than the number of reference directions ref_dirs={len(self.ref_dirs)}.\n"
                "This might cause unwanted behavior of the algorithm. \n"
                "Please make sure pop_size is equal or larger than the number of reference directions. ")
            
        survival = ReferenceDirectionSurvival(ref_dirs)

        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=survival,
                         eliminate_duplicates=eliminate_duplicates,
                         n_offsprings=n_offsprings,
                         output=output,
                         advance_after_initial_infill=True,
                         is_co_learn = is_co_learn,
                         is_do_learn = is_do_learn,
                         **kwargs)
    
    def _initialize(self):

        print("# Execution starts")

        super()._initialize()

        pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        pop.set("n_gen", self.n_gen)
        self.pop = pop
        self.evaluator.eval(self.problem, self.pop, algorithm=self)

        self.pop = self.survival.do(self.problem, self.pop, n_survive=len(self.pop), algorithm=self)

        """For CO"""
        if self.is_co_learn:
            self.child = []
            X = np.array([None]*self.problem.n_var)
            F = np.array([None]*self.problem.n_obj)
            self.co_target = np.array([None]*len(self.ref_dirs))
            for i in range (len(self.co_target)):
                self.co_target[i] = [X,F]
            self.co_frequency = 2 #since it is anyways adapted on-the-fly

        """For DO"""
        if self.is_do_learn:
            self.do_models = [None]*self.problem.n_obj
            self.termination_params = np.array([2, 20])
            dist = cdist(self.ref_dirs, self.ref_dirs)
            val = [np.sort(row)[1] for row in dist]
            self.last_termination = 1
            self.niche_radius = np.mean(val)
            self.survival_history = []
            self.niche_change = []
            self.pop_for_boundary_change = []
            self.do_trigger1 = False
            self.do_trigger2 = False
            self.do_frequency = 2 #since it is anyways adapted on-the-fly

        print("# Initialized successfully")

    def _infill(self):

        if self.n_gen % 25 == 0:
            print("# Generation: " + str(self.n_gen))

        if self.is_co_learn:
            self.history.append(self.pop)

        """Should we start CO?"""
        if self.is_co_learn and self.co_start:
            update_co_target(self)
            if self.n_gen>self.collection+1 and self.n_gen-self.last_co_repair>=self.co_frequency:
                # print('CO learning now')
                lower_bound,upper_bound, RF_model = co_learn(self)

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
            Q = self.pop.get("F")
            parent_niches = self.pop.get("niche")
            parent_niches = np.unique(parent_niches)

        """Call the Learn function"""
        if self.is_do_learn and self.do_trigger2 and self.n_gen-self.last_do_learn>=self.do_frequency:
            # print('DO learning now')
            do_learn(self)

        """Do the mating to produce the offspring self.off"""
        self.off = self.mating.do(self.problem, self.pop, self.n_offsprings, algorithm=self)

        self.off.set("n_gen", self.n_gen)
        if len(self.off) == 0:
            self.termination.force_termination = True
            return
        elif len(self.off) < self.n_offsprings:
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")

        """repair step:"""
        if self.n_gen>self.collection+1 and self.is_co_learn and self.n_gen-self.last_co_repair>=self.co_frequency and self.co_start:
            # print('CO repairing now')
            self.last_co_repair = self.n_gen
            co_repair(self, lower_bound,upper_bound, RF_model)

        """Call the Repair function and replace offspring"""
        if self.is_do_learn and self.do_trigger2 and self.n_gen-self.last_do_learn>=self.do_frequency:
            temp = [1 for model in self.do_models if model is None]
            if sum(temp)==0:
                # print('DO repairing now')
                self.last_do_learn = self.n_gen
                new_solutions = do_repair(self)
                sequence = np.arange(int(self.pop_size/2)) #DEXTER - offspring generated are already randomized - doesn't make sense to use another random input there
                for i in range(len(new_solutions)):
                    self.off[sequence[i]].X = np.array(new_solutions[i])

        self.evaluator.eval(self.problem, self.off, algorithm=self)
        if self.is_co_learn:
            self.child.append(self.off)

        self.Q = deepcopy(self.pop.get("F"))
        self.off_F = self.off.get("F")

        """Merge P_t and self.off into U_t"""
        self.pop = Population.merge(self.pop, self.off)

    def _advance(self, infills=None, **kwargs):

        # execute the survival to find the fittest solutions
        self.pop = self.survival.do(self.problem, self.pop, n_survive=self.pop_size, algorithm=self, **kwargs)

        """Update the archives required for stabilization tracking algorithm"""
        update_termination_archives(self, self.Q)

        """Check for termination"""
        if self.termination_suggestion[0] is None:
            term_cond = check_for_termination(self, 2, 20)
            if term_cond:
                self.termination_suggestion[0] = self.n_gen
                print("mild stabilization has reached at gen: "+str(self.n_gen))

        if self.termination_suggestion[1] is None:
            term_cond = check_for_termination(self, 3, 50)
            if term_cond:
                self.termination_suggestion = self.n_gen
                self.termination_pop = self.pop

        """Terminate, if suggested by the stabilization tracking algorithm"""
        if self.termination_suggestion[1] is not None:
            print("# The algorithm terminated after " + str(self.n_gen) + " generations.")
            self.termination.force_termination = True
            return
        
        """co_frequency adaptation"""
        df1 = self.pop.get("F")
        df2 = self.off_F[int(self.pop_size/2):]
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
        df2 = self.off_F[:int(self.pop_size/2)]
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
                termination_flag = check_for_termination(self, 2, 20)
                if termination_flag:
                    self.do_trigger2 = True
        
"""
Self-termination algorithm
"""

"""To update the archives related to the stabilization tracking algorithm"""
def update_termination_archives(self,Q):
    
    P = self.pop.get("F")
    nadir_point = self.pop[0].data['nadir_point']
    ideal_point = self.pop[0].data['ideal_point']
    if max(nadir_point) == np.inf:
        nadir_point = np.max(self.pop.get("F"), axis=0)

    if len(P)>0 and len(Q)>0:
        if nadir_point is not None:
            P_norm = normalize(P, xl=ideal_point, xu=nadir_point)
            Q_norm = normalize(Q, xl=ideal_point, xu=nadir_point)
        else:
            P_norm = np.array([(pt-ideal_point) for pt in P])
            Q_norm = np.array([(pt-ideal_point) for pt in Q])
        dist_matrix = calc_perpendicular_distance(P_norm, self.ref_dirs)
        assoc_P = np.array([np.argmin(row) for row in dist_matrix])
        dist_matrix = calc_perpendicular_distance(Q_norm, self.ref_dirs)
        assoc_Q = np.array([np.argmin(row) for row in dist_matrix])
        mu_D = 0
        count = 0
        for i in range (len(self.ref_dirs)):
            cluster_P = [p for p in range(len(assoc_P)) if assoc_P[p]==i]
            cluster_Q = [p for p in range(len(assoc_Q)) if assoc_Q[p]==i]
            if len(cluster_P)>0 and len(cluster_Q)>0:
                p = np.mean(P_norm[cluster_P], axis=0)
                q = np.mean(Q_norm[cluster_Q], axis=0)
                D = abs(np.matmul(self.ref_dirs[i],p)-np.matmul(self.ref_dirs[i],q))
                if D!=0:
                    D = D/max(np.matmul(self.ref_dirs[i],p),np.matmul(self.ref_dirs[i],q))
                else:
                    D = 0
                mu_D += D
                count += 1
            elif  len(cluster_P)==0 and len(cluster_Q)>0:
                mu_D += 1
                count += 1
            elif  len(cluster_P)>0 and len(cluster_Q)==0:
                mu_D += 1
                count += 1
        mu_D = mu_D/count
    else:
        mu_D = 1
    self.mu_D.append(mu_D)
    self.D_t.append(np.mean(self.mu_D))
    self.S_t.append(np.std(self.mu_D))

"""Check the conditions for termination"""
def check_for_termination(self,n_p,ns):
    to_terminate = False
    if self.n_gen>ns:
        D_t = self.D_t[-ns:]
        S_t = self.S_t[-ns:]
        D_t = [round(val,n_p) for val in D_t]
        S_t = [round(val,n_p) for val in S_t]
        if len(np.unique(D_t))==1 and len(np.unique(S_t))==1:
            to_terminate = True
        return to_terminate

# =========================================================================================================
# Survival
# =========================================================================================================


class ReferenceDirectionSurvival(Survival):

    def __init__(self, ref_dirs):
        super().__init__(filter_infeasible=True)
        self.ref_dirs = ref_dirs
        self.opt = None
        self.norm = HyperplaneNormalization(ref_dirs.shape[1])

    def _do(self, problem, pop, n_survive, D=None, **kwargs):

        # attributes to be set after the survival
        F = pop.get("F")

        # calculate the fronts of the population
        fronts, rank = NonDominatedSorting().do(F, return_rank=True, n_stop_if_ranked=n_survive)
        non_dominated, last_front = fronts[0], fronts[-1]

        # update the hyperplane based boundary estimation
        hyp_norm = self.norm
        hyp_norm.update(F, nds=non_dominated)
        ideal, nadir = hyp_norm.ideal_point, hyp_norm.nadir_point

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

        # associate individuals to niches
        niche_of_individuals, dist_to_niche, dist_matrix = \
            associate_to_niches(F, self.ref_dirs, ideal, nadir)

        # attributes of a population
        pop.set('rank', rank,
                'niche', niche_of_individuals,
                'dist_to_niche', dist_to_niche,
                'ideal_point', [ideal for row in range(len(pop))],
                'nadir_point', [nadir for row in range(len(pop))],
                )

        # set the optimum, first front and closest to all reference directions
        closest = np.unique(dist_matrix[:, np.unique(niche_of_individuals)].argmin(axis=0))
        self.opt = pop[intersect(fronts[0], closest)]
        if len(self.opt) == 0:
            self.opt = pop[fronts[0]]

        # if we need to select individuals to survive
        if len(pop) > n_survive:

            # if there is only one front
            if len(fronts) == 1:
                n_remaining = n_survive
                until_last_front = np.array([], dtype=int)
                niche_count = np.zeros(len(self.ref_dirs), dtype=int)

            # if some individuals already survived
            else:
                until_last_front = np.concatenate(fronts[:-1])
                niche_count = calc_niche_count(len(self.ref_dirs), niche_of_individuals[until_last_front])
                n_remaining = n_survive - len(until_last_front)

            S = niching(pop[last_front], n_remaining, niche_count, niche_of_individuals[last_front],
                        dist_to_niche[last_front])

            survivors = np.concatenate((until_last_front, last_front[S].tolist()))
            pop = pop[survivors]

        return pop


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
    niche_count = np.zeros(n_niches, dtype=int)
    index, count = np.unique(niche_of_individuals, return_counts=True)
    niche_count[index] = count
    return niche_count


# =========================================================================================================
# Normalization
# =========================================================================================================


class HyperplaneNormalization:

    def __init__(self, n_dim) -> None:
        super().__init__()
        self.ideal_point = np.full(n_dim, np.inf)
        self.worst_point = np.full(n_dim, -np.inf)
        self.nadir_point = None
        self.extreme_points = None

    def update(self, F, nds=None):
        # find or usually update the new ideal point - from feasible solutions
        self.ideal_point = np.min(np.vstack((self.ideal_point, F)), axis=0)
        self.worst_point = np.max(np.vstack((self.worst_point, F)), axis=0)

        # this decides whether only non-dominated points or all points are used to determine the extreme points
        if nds is None:
            nds = np.arange(len(F))

        # find the extreme points for normalization
        self.extreme_points = get_extreme_points_c(F[nds, :], self.ideal_point,
                                                   extreme_points=self.extreme_points)

        # find the intercepts for normalization and do backup if gaussian elimination fails
        worst_of_population = np.max(F, axis=0)
        worst_of_front = np.max(F[nds, :], axis=0)

        self.nadir_point = get_nadir_point(self.extreme_points, self.ideal_point, self.worst_point,
                                           worst_of_front, worst_of_population)


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

def co_learn(self):
    obj_data = []
    design_data = []
    input_data = []
    output_data = []
    for i in range(1,self.collection+1):
        temp = self.child[-i]
        for row in temp:
            obj_data.append(row.F)
            design_data.append(row.X)

    """Adding the first parent pop to the archive"""
    for row in self.history[-(self.collection+1)]:
        obj_data.append(row.F)
        design_data.append(row.X)

    """Normalize and map the archive points to the targets"""
    nadir_point = self.pop[0].data['nadir_point']
    ideal_point = self.pop[0].data['ideal_point']
    if max(nadir_point) == math.inf:
        nadir_point = np.max(self.pop.get("F"), axis=0)
    obj_data = normalize(np.array(obj_data), xl = np.array(ideal_point)-1e-12, xu = np.array(nadir_point))
    for i in range(len(design_data)):
        temp = calc_perpendicular_distance([obj_data[i]],self.ref_dirs)[0]
        ref_index = np.argmin(temp)
        if self.co_target[ref_index][0][0] is not None:
            input_data.append(design_data[i])
            output_data.append(deepcopy(self.co_target[ref_index][0]))

    input_data = np.array(input_data)
    output_data = np.array(output_data)

    input_copy = []
    output_copy = []
    
    dup_index = []
    for i in range (len(input_data)):
        flag = False
        for j in range(len(input_data[i])):
            if math.isnan(input_data[i, j]) or math.isnan(output_data[i, j]):
                flag = True
        if np.linalg.norm(output_data[i]-input_data[i])==0 or flag==True:
            dup_index.append(i)
        else:
            input_copy.append(input_data[i][:])
            output_copy.append(output_data[i][:])
    input_data = np.array([val for val in input_copy])
    output_data = np.array([val for val in output_copy])

    for i in range(len(input_data)):
        for j in range(len(input_data[i])):
            if math.isnan(input_data[i, j]):
                print('NaN in input')
    for i in range(len(output_data)):
        for j in range(len(output_data[i])):
            if math.isnan(output_data[i, j]):
                print('NaN in output')
    
    minbound = np.min([np.min(input_data,axis=0),np.min(output_data,axis=0)],axis=0)
    maxbound = np.max([np.max(input_data,axis=0),np.max(output_data,axis=0)],axis=0)
    lower_bound = np.mean([self.problem.xl,minbound],axis=0)
    upper_bound = np.mean([self.problem.xu,maxbound],axis=0)
    input_data = normalize(input_data,xl=lower_bound,xu=upper_bound)
    output_data = normalize(output_data,xl=lower_bound,xu=upper_bound)
    regr = RandomForestRegressor(max_features=self.problem.n_var, min_samples_split=2, random_state=1, n_estimators=int(self.collection*self.pop_size))
    regr.fit(input_data,output_data)

    return lower_bound,upper_bound, regr

def co_boundary_repair(p,c,l,u,iters):
    p[p==c] = (u+l)[p==c]/2
    normv = np.linalg.norm(p-c)
    idl,idr = c<l,u<c
    if sum(idl+idr)==0:
        data = c
    else:
        d = normv*np.max([idl*(l-c)/(p-c),idr*(u-c)/(p-c)],axis=0)
        alpha = (normv-d)/normv
        up = np.array([~idl*((l-c)/(p-c)), ~idr*(u-c)/(p-c)])
        D = normv*min(up[up>0])
        r = np.random.random()
        if r==0:
            Y = d
        else:
            atan = (D-d)/(alpha*d)
            atan = np.array([math.tan(r*math.atan(val)) for val in atan])
            Y = d*(1+alpha*atan)
            data = c + (p-c)*Y/normv
    idl,idr = data<l,u<data
    if sum(idl+idr)>0:
        print(iters)
        if iters<10:
            co_boundary_repair(p,data,l,u,iters+1)
        else:
            print(p,c,data)
            for i in range(len(data)):
                if data[i] < l[i] or data[i] > u[i]:
                    data[i] = l[i] + np.random.random()*abs(u[i] - l[i])
    return np.array(data)

def boundary_repair(old, new, problem):
    for i in range(len(new)):
        if new[i]>problem.xu[i]:
            new[i] = old[i] + np.random.random()*abs(problem.xu[i]-old[i])
        elif new[i]<problem.xl[i]:
            new[i] = problem.xl[i] + np.random.random()*abs(old[i]-problem.xl[i])
    return new

def co_repair(self, lower_bound, upper_bound, regr):
    child = np.array([self.off[i].X for i in range (self.pop_size)])
    child = normalize(child,xl=lower_bound,xu=upper_bound)
    child = regr.predict(child)
    child = denormalize(child,lower_bound,upper_bound)
    for i in range (0,int(self.pop_size*self.co_repair_fraction)):
        # index = np.random.randint(0,self.pop_size-1)
        index = i + int(self.pop_size/2) #DEXTER - since the mating is random, doesn't make sense to involve another random selection process
        self.co_boost = 1 + 0.5*np.random.random()
        modification = np.array(self.co_boost*(np.array(child[index])-self.off[index].X))
        original_child = deepcopy(self.off[index].X)
        self.off[index].X += modification

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
                self.off[index].X[j] -= 1.0*(original_child[j])
            
        """Variable Boundary Repair"""
        self.off[index].X = co_boundary_repair(original_child,self.off[index].X,self.problem.xl,self.problem.xu,0)
        # self.off[index].X = boundary_repair(original_child, self.off[index].X, self.problem)

def update_co_target(self):
    """update target-dataset: [X,F]*pop_size"""
    X = [self.pop[i].X for i in range (self.pop_size)]
    F = [self.pop[i].F for i in range (self.pop_size)]

    """target normalization as per current pop"""
    nadir_point = self.pop[0].data['nadir_point']
    ideal_point = self.pop[0].data['ideal_point']
    if max(nadir_point) == math.inf:
        nadir_point = np.max(self.pop.get("F"), axis=0)
    F_norm = normalize(np.array(F), xl = ideal_point-1e-12, xu = nadir_point)
    assoc_index = []
    assoc_value = []
    for obj in F_norm:
        temp = calc_perpendicular_distance([obj],self.ref_dirs)[0]
        assoc_index.append(np.argmin(temp))
        assoc_value.append(min(temp))

    """check if a population member can/should replace a target"""
    for i in range(self.pop_size):
        index = assoc_index[i]
        if self.co_target[index][0][0] is None:
            self.co_target[index][0] = X[i][:]
            self.co_target[index][1] = F[i][:]
        else:
            target_F = normalize(self.co_target[index][1], xl = ideal_point-1e-12, xu = nadir_point)
            if sum(target_F > F_norm[i]) >= 1 and sum(target_F >= F_norm[i]) == self.problem.n_obj:
                self.co_target[index][0] = X[i][:]
                self.co_target[index][1] = F[i][:]
            elif sum(target_F < F_norm[i]) >= 1 and sum(target_F > F_norm[i]) >= 1:
                target_perp = calc_perpendicular_distance([target_F],[self.ref_dirs[index]])[0][0]
                if assoc_value[i] < target_perp:
                    self.co_target[index][0] = X[i][:]
                    self.co_target[index][1] = F[i][:]

def do_predict(X, regr, problem, eta, category, niche_radius):
    X = normalize(X, xl=problem.xl, xu=problem.xu)
    X_ = regr.predict([X])[0]
    X_ = denormalize(X_, xl=problem.xl, xu=problem.xu)
    if category =='pit':
        eta = (-0.5+np.random.random())+eta
    else:
        eta = (np.random.random())*eta*(2**0.5)/niche_radius
    X_ = X + eta*(X_)
    X_ = boundary_repair(X, X_, problem)
    # X_ = co_boundary_repair(X, X_, problem.xl, problem.xu, 0)
    return X_

def do_learn(self): 
    
    """Extract the data, normalize objectives"""
    # X = self.pop.get("X")
    # F = self.pop.get("F")
    X = np.array([ind.X for ind in self.pop if ind.data["rank"]==0]) 
    F = np.array([ind.F for ind in self.pop if ind.data["rank"]==0]) 
    nadir_point = self.pop[0].data['nadir_point']
    ideal_point = self.pop[0].data['ideal_point']
    if max(nadir_point) == math.inf:
        nadir_point = np.max(self.pop.get("F"), axis=0)
    F_norm = normalize(F, xl=ideal_point-1e-12, xu=nadir_point)
    F_norm = np.array([row/sum(row) for row in F_norm])

    """Set the range as per granularity"""
    min_d = 0.5*self.niche_radius
    max_d = 1.5*self.niche_radius

    """For each objective, learn an ML model"""
    for i in range(self.problem.n_obj):

        """Generate the training dataset"""
        input_data = []
        output_data = []
        for j in range(len(F)):
            """Find the solutions between 0.5*r and 1.5*r (closed bracket)"""
            indices = [k for k in range(len(F)) if euclidean(self.ref_dirs[self.pop[j].data['niche']], F_norm[k])>=min_d and euclidean(self.ref_dirs[self.pop[j].data['niche']], F_norm[k])<=max_d]
            # """Find the solutions between 0.5*r and 1.5*r (closed bracket)"""
            # indices = [k for k in range(len(F)) if euclidean(F_norm[j], F_norm[k])>=min_d and euclidean(F_norm[j], F_norm[k])<=max_d]
            # """Make sure the solutions belong to some other RV"""
            # indices = [index for index in indices if self.pop[j].data['niche']!=self.pop[index].data['niche']]
            if len(indices)>0:
                """calculate the improvement"""
                improvement = [(F_norm[index, i]-F_norm[j, i]) for index in indices]
                """use the solution that offers best improvement (max. negative)"""
                if min(improvement)<0:
                    index = np.argmin(improvement)
                    """only a movement of 'r' is to be learnt in F-space"""
                    input_data.append(X[j])
                    output_data.append((X[indices[index]]-X[j]))
        
        input_data = np.array(input_data)
        output_data = np.array(output_data)

        """train the ML model"""
        if len(input_data)>0:
            input_data = normalize(input_data, xl=self.problem.xl, xu=self.problem.xu)
            output_data = normalize(output_data, xl=self.problem.xl, xu=self.problem.xu)
            self.do_models[i] = KNeighborsRegressor(n_neighbors=min(len(input_data), max(self.problem.n_obj, self.problem.n_var)))
            # self.do_models[i] = RandomForestRegressor(max_features=self.problem.n_var, min_samples_split=2, random_state=1, n_estimators=int(self.pop_size))
            self.do_models[i].fit(input_data, output_data)

def do_repair(self):
    """extract the data, normalize objectives"""
    X = self.pop.get("X")
    F = self.pop.get("F") 
    nadir_point = self.pop[0].data['nadir_point']
    ideal_point = self.pop[0].data['ideal_point']
    if max(nadir_point) == math.inf:
        nadir_point = np.max(self.pop.get("F"), axis=0)
    F_norm = normalize(F, xl=ideal_point-1e-12, xu=nadir_point)
    F_norm = np.array([row/sum(row) for row in F_norm])

    """Identify the pits"""
    niches = np.array([ind.data["niche"] for ind in self.pop])
    no_of_pits = len(self.ref_dirs) - len(np.unique(niches))
    pit_indices = np.array([i for i in range(len(self.ref_dirs)) if i not in niches])

    """Identify the boundary RVs with at least one associated solution"""
    boundary_RVs = np.array([i for i in range(len(self.ref_dirs)) if min(self.ref_dirs[i])==0 and i in niches])

    """Set the repair fraction for the first time"""
    if self.do_repair_fraction is None:
        # Rg = self.min_repair[0] + np.random.random()*(self.max_repair[0]-self.min_repair[0])
        # Rb = self.min_repair[1] + np.random.random()*(self.max_repair[1]-self.min_repair[1])
        # self.do_repair_fraction = [Rg, Rb]
        self.do_repair_fraction = [0.25, 0.25]

    """Choose the starting points"""
    n_repair_pits = int(self.do_repair_fraction[0]*self.pop_size) # for the pits
    n_repair_boundary = int(self.do_repair_fraction[1]*self.pop_size) # for the boundary

    new_solutions = []

    """Repair the pit solutions"""
    if n_repair_pits > 0 and no_of_pits>0:
        if no_of_pits <= n_repair_pits:
            temp_indices = []
            while len(temp_indices)<n_repair_pits:
                temp_indices.extend(np.random.permutation(pit_indices))
        else:
            temp_indices = np.random.permutation(pit_indices)
        for i in range(n_repair_pits):
            pit = self.ref_dirs[temp_indices[i]]
            dist = calc_perpendicular_distance(F_norm, [pit])
            nearest_index = np.argmin([row[0] for row in dist])
            nearest_distance = min([row[0] for row in dist])
            starting_point = X[nearest_index]
            """find in which objective max. increment/reduction is needed"""
            diff_vector = pit-F_norm[nearest_index]
            abs_diff_vector = np.array([abs(row) for row in diff_vector])
            index = np.argmax(abs_diff_vector)
            if diff_vector[index]>0:
                index = index + self.problem.n_obj
            if nearest_distance<0.5*self.niche_radius or nearest_distance>1.5*self.niche_radius:
                progress_length = np.linalg.norm(diff_vector)/self.niche_radius
            else:
                progress_length = 1
            if index<self.problem.n_obj:
                temp = do_predict(starting_point, self.do_models[index], self.problem, progress_length, 'pit', self.niche_radius)
            else:
                temp = do_predict(starting_point, self.do_models[index-self.problem.n_obj], self.problem, -1*progress_length, 'pit', self.niche_radius)
            new_solutions.append(temp)

    """Repair the boundary solutions"""
    if n_repair_boundary > 0 and len(boundary_RVs) > 0:
        if len(boundary_RVs) <= n_repair_boundary:
            temp_indices = []
            while len(temp_indices)<n_repair_boundary:
                temp_indices.extend(np.random.permutation(boundary_RVs))
        else:
            temp_indices = np.random.permutation(boundary_RVs)
        for i in range(n_repair_boundary):
            index_RV = temp_indices[i] 
            RV = self.ref_dirs[index_RV]
            nearest_index = np.where(niches==index_RV)[0]
            if len(nearest_index)>1:
                nearest_index = nearest_index[np.random.choice(len(nearest_index))]
            else:
                nearest_index = nearest_index[0]
            starting_point = X[nearest_index]
            index_model = np.where(RV==0)[0]
            if len(index_model)>1:
                index_model = index_model[np.random.choice(len(index_model))]
            else:
                index_model = index_model[0]
            if self.problem.n_obj==2:
                temp = do_predict(starting_point, self.do_models[index_model], self.problem, 1, 'boundary', self.niche_radius)
            else:
                temp = do_predict(starting_point, self.do_models[index_model], self.problem, 1, 'boundary', self.niche_radius)
            new_solutions.append(temp)

    for i in range(len(new_solutions)):
        for j in range(len(new_solutions[i])):
            if math.isnan(new_solutions[i][j]):
                print('NaN in DO-offspring')
    
    return new_solutions
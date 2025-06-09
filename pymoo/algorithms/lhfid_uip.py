import numpy as np
from scipy.spatial.distance import cdist
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


# =========================================================================================================
# Implementation
# =========================================================================================================

class LHFiD(GeneticAlgorithm):

    def __init__(self,
                 ref_dirs,
                 n_neighbors=20,
                 decomposition='auto',
                 prob_neighbor_mating=0.9,
                 display=MultiObjectiveDisplay(),
                 **kwargs):
        """

        LHFiD Algorithm.

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
        self.termination_suggestion = [None,None,None] # save the generations at which termination is suggested
        self.termination_pop = [None,None,None] # save the population when the termination is suggested

        # initialize the termination archive as empty set.
        self.mu_D = []
        self.D_t = []
        self.S_t = []

        set_if_none(kwargs, 'pop_size', len(ref_dirs))
        set_if_none(kwargs, 'sampling', FloatRandomSampling())
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob=1.0, eta=20))
        set_if_none(kwargs, 'mutation', PolynomialMutation(prob=None, eta=20))
        set_if_none(kwargs, 'survival', None)
        set_if_none(kwargs, 'selection', TournamentSelection(func_comp=comp_by_cv_then_random))

        super().__init__(display=display, **kwargs)

        # initialized when problem is known
        self.ref_dirs = ref_dirs
    
    def _initialize(self):

        super()._initialize()
        self.ideal_point = np.min(self.pop.get("F"), axis=0)

    def _next(self):

        print(self.co_frequency, self.do_frequency)

        #set to termination
        self.min_constr_value = min([ind.CV for ind in self.pop])

        # check = [val for val in self.termination_suggestion if val is None]
        # if len(check)==0 and self.is_co_learn and self.is_do_learn:
        # # if len(check)==0:
        #     self.termination.force_termination = True
        #     return

        # print(self.n_gen)
        repair, crossover, mutation = self.repair, self.mating.crossover, self.mating.mutation

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

        """Call the Learn function"""
        if self.is_do_learn and self.do_trigger2 and self.n_gen-self.last_do_learn>=self.do_frequency:
            print('DO learning now')
            GeneticAlgorithm.do_learn(self)

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

        """repair step:"""
        if self.n_gen>self.collection+1 and self.is_co_learn and self.n_gen-self.last_co_repair>=self.co_frequency and self.co_start:
            print('CO repairing now')
            self.last_co_repair = self.n_gen
            GeneticAlgorithm.co_repair(self, lower_bound,upper_bound, RF_model)

        """Call the Repair function and replace offspring"""
        if self.is_do_learn and self.do_trigger2 and self.n_gen-self.last_do_learn>=self.do_frequency:
            temp = [1 for model in self.do_models if model is None]
            if sum(temp)==0:
                print('DO repairing now')
                self.last_do_learn = self.n_gen
                new_solutions = GeneticAlgorithm.do_repair(self)
                sequence = np.arange(int(self.pop_size/2)) #DEXTER - offspring generated are already randomized - doesn't make sense to use another random input there
                for i in range(len(new_solutions)):
                    self.off[sequence[i]].X = np.array(new_solutions[i])

        # evaluate the offspring
        self.evaluator.eval(self.problem, self.off, algorithm=self)

        if self.is_co_learn:
            self.child.append(self.off)

        # merge the offsprings with the current population
        self.pop = Population.merge(self.pop, self.off)

        # initialize the feasible and infeasible population
        self.feas_index, self.infeas_index = Population(), Population()

        # split feasible and infeasible solutions - only indices are transferred
        self.feas_index, self.infeas_index = split_by_feasibility(self.pop, sort_infeasbible_by_cv=True) 

        # the do survival selection
        self.pop = survival_selection(self)

        """Update termination archives"""
        if self.do_trigger1==True:
            update_termination_archives(self, Q_term)

        #identify/update the nadir point
        # self.nadir_point = get_nadir_point(self)
        check_for_nadir_update(self,2,20) # parameters for nadir-point update

        """co_frequency adaptation"""
        df1 = self.pop.get("F")
        df2 = self.off.get("F")[int(self.pop_size/2):]
        count = 0
        for val in df1:
            if sum(sum(df2==val))>0:
                count+=1
        self.co_survived.append(count)
        if self.n_gen>self.collection+1 and self.is_co_learn and self.n_gen-self.last_co_repair==0 and self.co_start:
            if self.co_survived[-1]>self.co_survived[-2] and self.co_frequency>2 and self.do_trigger2:
                self.co_frequency-=1
            elif self.co_survived[-1]>self.co_survived[-2] and self.co_frequency>1 and self.do_trigger2==False:
                self.co_frequency-=1
            elif self.co_survived[-1]<self.co_survived[-2]:
                self.co_frequency+=1

        """do_frequency adaptation"""
        df1 = self.pop.get("F")
        df2 = self.off.get("F")[:int(self.pop_size/2)]
        count = 0
        for val in df1:
            if sum(sum(df2==val))>0:
                count+=1
        self.do_survived.append(count)
        if self.is_do_learn and self.do_trigger2 and self.n_gen-self.last_do_learn==0:
            temp = [1 for model in self.do_models if model is None]
            if sum(temp)==0:
                if self.do_survived[-1]>self.do_survived[-2] and self.do_frequency>2 and self.co_start:
                    self.do_frequency-=1
                elif self.do_survived[-1]>self.do_survived[-2] and self.do_frequency>1 and self.co_start==False:
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

        #check global termination
        CV = self.pop.get("CV")
        # print(min(CV))
        if min(CV)==0:
            if self.termination_suggestion[0] is None:
                term_cond = check_for_termination(self,2,20) # termination-2 parameters
                if term_cond:
                    self.termination_suggestion[0] = self.n_gen
                    self.termination_pop[0] = self.pop

            if self.termination_suggestion[1] is None:
                term_cond = check_for_termination(self,3,20) # termination-2 parameters
                if term_cond:
                    self.termination_suggestion[1] = self.n_gen
                    self.termination_pop[1] = self.pop
            
            if self.termination_suggestion[2] is None:
                term_cond = check_for_termination(self,3,50) # termination-2 parameters
                if term_cond:
                    self.termination_suggestion[2] = self.n_gen
                    self.termination_pop[2] = self.pop

def survival_selection(self):

    # print(len(self.pop))

    #constraint-handling
    if self.problem.n_constr > 0:
        feas_pop = self.pop[self.feas_index]
        infeas_pop = self.pop[self.infeas_index]
        if len(feas_pop)==0:
            CV = np.array([abs(ind.CV) for ind in infeas_pop])
            selection_index = np.argsort(CV)[:self.pop_size]
            survivors = infeas_pop[selection_index]
            survivors = np.array([ind[0] for ind in survivors])
            # print(infeas_pop.shape, survivors.shape)
            pop = pop_from_array_or_individual(survivors[0])
            for i in range(1,len(survivors)):
                temp = pop_from_array_or_individual(survivors[i])
                pop = Population.merge(pop,temp)
            return pop
        elif len(feas_pop)<=self.pop_size:
            feas_survivors = feas_pop
            n_survive = self.pop_size - len(feas_survivors)
            CV = np.array([abs(ind.CV) for ind in infeas_pop])
            selection_index = np.argsort(CV)[:n_survive]
            infeas_survivors = infeas_pop[selection_index]
            infeas_survivors = np.array([ind[0] for ind in infeas_survivors])
            pop = pop_from_array_or_individual(feas_survivors[0])
            for i in range(1,len(feas_survivors)):
                temp = pop_from_array_or_individual(feas_survivors[i])
                pop = Population.merge(pop,temp)
            for i in range(len(infeas_survivors)):
                temp = pop_from_array_or_individual(infeas_survivors[i])
                pop = Population.merge(pop,temp)
            # print('merging feas-infeas',len(pop))
            return pop
        else:
            F = np.array([ind.F for ind in feas_pop])
    
    else:
        F = self.pop.get("F")
        feas_pop = self.pop

    self.ideal_point = np.min(np.vstack((self.ideal_point, F)), axis=0)
    if self.nadir_point is None:
        F_norm = np.array([(pt-self.ideal_point) for pt in F])
    else:
        F_norm = normalize(F, x_min=self.ideal_point, x_max=self.nadir_point)
    dist_matrix = load_function('calc_perpendicular_distance')(F_norm, self.ref_dirs)
    assoc_index = np.array([np.argmin(row) for row in dist_matrix])
    assoc_value = np.array([min(row) for row in dist_matrix])

    feas_pop.set('rank', np.ones(len(feas_pop)),
            'niche', assoc_index,
            'dist_to_niche', assoc_value,#)
            'ideal_point', np.array([self.ideal_point]*len(feas_pop)), #DEXTER
            'nadir_point', np.array([self.nadir_point]*len(feas_pop))) #DEXTER

    # initialize survivors as empty - store the indices of solutions
    survivors = []
    selected_vectors = []
    data = [] #insights

    # first-filling of solutions - for non-empty clusters
    for i in range (len(self.ref_dirs)):

        cluster_indices = [p for p in range(len(assoc_index)) if assoc_index[p]==i]
        cluster_perp_values = [assoc_value[p] for p in range(len(assoc_index)) if assoc_index[p]==i]
        if len(cluster_indices)==1:
            feas_pop[cluster_indices[0]].set('rank', 0)
            survivors.append(cluster_indices[0])
            selected_vectors.append(i)
            data.append(1) #insights
        elif len(cluster_indices)>1:
            
            # check for pareto-dominance
            is_pareto_non_dom = []
            for ind1 in cluster_indices:
                count = 0
                for ind2 in cluster_indices:
                    if ind1 != ind2:    
                        n_b = sum(F_norm[ind1]>F_norm[ind2])
                        n_w = sum(F_norm[ind1]<F_norm[ind2])
                        if n_b>=1 and n_w==0:
                            count+=1
                if count==0:
                    is_pareto_non_dom.append(True)
                    feas_pop[ind1].set('rank', 0)
                else:
                    is_pareto_non_dom.append(False)

            # form the cluster with only pareto-non-dominated solutions
            cluster_indices = [cluster_indices[p] for p in range(len(cluster_indices)) if is_pareto_non_dom[p] is True]
            cluster_perp_values = [cluster_perp_values[p] for p in range(len(cluster_perp_values)) if is_pareto_non_dom[p] is True]
            alpha_solution = cluster_indices[np.argmin(cluster_perp_values)] #absolute indexing

            # find all candidates that lhf-dominate alpha
            if max(self.ref_dirs[i])<1: #doing this comparison only for non-axis reference-vectors
                candidates = []
                candidates_perp_d = []
                for p in range(len(cluster_indices)):
                    ind = cluster_indices[p]
                    if ind != alpha_solution:
                        temp_alpha, temp_ind = F_norm[alpha_solution], F_norm[ind]
                        for q in range(self.problem.n_obj):
                            if abs(temp_alpha[q]-temp_ind[q])<1e-06:
                                temp_alpha[q] = temp_ind[q]
                        n_b = sum(temp_alpha>temp_ind)
                        n_w = sum(temp_alpha<temp_ind)
                        # n_b = sum(F_norm[alpha_solution]>F_norm[ind])
                        # n_w = sum(F_norm[alpha_solution]<F_norm[ind])
                        # if n_b >= n_w:
                        if n_b > n_w: #trial only
                            decide = tie_breaker(F_norm[ind], F_norm[alpha_solution], self.ref_dirs[i])
                            if decide<0:
                                candidates.append(ind)
                                candidates_perp_d.append(cluster_perp_values[p])
                # choose the solution that offers best delta_f value
                if len(candidates)>0:
                    alpha_solution = candidates[np.argmin(candidates_perp_d)]
                    data.append(2) #insights
                else:
                    data.append(3) #insights
            else:
                data.append(3) #insights
            survivors.append(alpha_solution)
            selected_vectors.append(i)

    self.data = data #insights

    # second filling of solutions near the empty vectors
    left_solutions = [i for i in range(len(feas_pop)) if i not in survivors]
    left_vectors = [i for i in range(len(self.ref_dirs)) if i not in selected_vectors]

    #choosing the solutions closest to the left-over reference-vectors.
    dist_matrix = load_function('calc_perpendicular_distance')(F_norm[left_solutions], self.ref_dirs[left_vectors])
    for i in range(len(left_vectors)):
        dis = dist_matrix[:,i]
        survivors.append(left_solutions[np.argmin(dis)])

    # if len(survivors)==self.pop_size:
    return feas_pop[survivors]

#takes in F-vector of alpha-solution, the solution in comparison and the associated ref-vector
def tie_breaker(f, alpha, rd):
    M = len(rd)
    sum_val = 0
    for i in range(M):
        if rd[i]==0:
            temp = 1e-6
        else:
            temp = rd[i]
        sum_val += (f[i] - alpha[i])/temp
##        print('RVs')
        # rd = [1, 1, 1]
        # print(rd)
        # sum_val += (f[i] - alpha[i])*rd[i]
    return sum_val

#termination function
def update_termination_archives(self,Q):

    #normalization
    P, CV = self.pop.get("F"), self.pop.get("CV")
    # print(P, self.pop)
    P = np.array([P[i] for i in range(self.pop_size) if CV[i]==0]) # P has only feasible solutions

    if len(P)>0 and len(Q)>0:
        if self.nadir_point is None:
            P_norm = np.array([(pt-self.ideal_point) for pt in P])
            Q_norm = np.array([(pt-self.ideal_point) for pt in Q])
        else:
            P_norm = normalize(P, x_min=self.ideal_point, x_max=self.nadir_point)
            Q_norm = normalize(Q, x_min=self.ideal_point, x_max=self.nadir_point)

        # association with ref-vectors
        dist_matrix = load_function('calc_perpendicular_distance')(P_norm, self.ref_dirs)
        assoc_P = np.array([np.argmin(row) for row in dist_matrix])
        dist_matrix = load_function('calc_perpendicular_distance')(Q_norm, self.ref_dirs)
        assoc_Q = np.array([np.argmin(row) for row in dist_matrix])

        # main termination loop for each reference-vector
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

    #update the archives
    self.mu_D.append(mu_D)
    self.D_t.append(np.mean(self.mu_D))
    self.S_t.append(np.std(self.mu_D))

    #check conditions
def check_for_nadir_update(self,n_p,ns):

    self.term_gen += 1
    if self.term_gen>ns:
        D_t = self.D_t[-ns:]
        S_t = self.S_t[-ns:]
        D_t = [round(val,n_p) for val in D_t]
        S_t = [round(val,n_p) for val in S_t]
        if len(np.unique(D_t))==1 and len(np.unique(S_t))==1:
            # self.term_gen = 1 # doing it only if the hyperplane makes sense
            self.nadir_point = get_nadir_point(self)
            # self.nadir_point = np.max(self.pop.get("F"), axis=0)
            print('nadir update: ',self.n_gen)

def check_for_termination(self,n_p,ns):

    to_terminate = False
    if len(self.D_t)>ns:# and max(self.min_constr_value[-ns:])==0:
        D_t = self.D_t[-ns:]
        S_t = self.S_t[-ns:]
        D_t = [round(val,n_p) for val in D_t]
        S_t = [round(val,n_p) for val in S_t]
        if len(np.unique(D_t))==1 and len(np.unique(S_t))==1:
            to_terminate = True
        return to_terminate

def get_nadir_point(self):

    ideal_point = self.ideal_point
    F, CV = self.pop.get("F"), self.pop.get("CV")
    F = np.array([F[i] for i in range(self.pop_size) if CV[i]==0]) # F has only feasible solutions

    # check for pareto-dominance
    is_pareto_non_dom = []
    for ind1 in F:
        count = 0
        for ind2 in F:    
            n_b = sum(ind1>ind2)
            n_w = sum(ind1<ind2)
            if n_b>=1 and n_w==0:
                count+=1
        if count==0:
            is_pareto_non_dom.append(True)
        else:
            is_pareto_non_dom.append(False)

    F = np.array([F[i] for i in range(len(F)) if is_pareto_non_dom[i]==True])

    if len(F)==0:
        return None
    else:
        self.extreme_points = get_extreme_points_c(F,ideal_point,self.extreme_points)
        extreme_points = self.extreme_points

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
            else:
                self.term_gen = 1

        except LinAlgError:

            # fall back to the earlier nadir-point, to try again in next generation
            nadir_point = self.nadir_point

        return nadir_point

def get_extreme_points_c(F, ideal_point, extreme_points):
    # calculate the asf which is used for the extreme point decomposition
    weights = np.eye(F.shape[1])
    weights[weights == 0] = 1e6

    # add the old extreme points to never loose them for normalization
    _F = F
    if extreme_points is not None:
        _F = np.concatenate([extreme_points, _F], axis=0)

    # use __F because we substitute small values to be 0
    __F = _F - ideal_point
    __F[__F < 1e-3] = 0 # just to see the effect in WFG2,6 (M=8)

    # update the extreme points for the normalization having the highest asf value each
    F_asf = np.max(__F * weights[:, None, :], axis=2)

    I = np.argmin(F_asf, axis=1)
    extreme_points = _F[I, :]

    return extreme_points

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

def split_by_feasibility(pop, sort_infeasbible_by_cv=True):
    CV = pop.get("CV")

    b = (CV <= 0)

    feasible = np.where(b)[0]
    infeasible = np.where(np.logical_not(b))[0]

    if sort_infeasbible_by_cv:
        infeasible = infeasible[np.argsort(CV[infeasible, 0])]

    return feasible, infeasible

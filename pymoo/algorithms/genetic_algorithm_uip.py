from pymoo.model.algorithm import Algorithm
from pymoo.model.duplicate import DefaultDuplicateElimination, NoDuplicateElimination
from pymoo.model.individual import Individual
from pymoo.model.initialization import Initialization
from pymoo.model.mating import Mating
from pymoo.model.population import Population
from pymoo.model.repair import NoRepair
# from scipy._lib.six import b

"""New imports"""
from pymoo.factory import get_reference_directions
import numpy as np 
from pymoo.util.normalization import normalize, denormalize
from pymoo.util.function_loader import load_function
from copy import deepcopy
from scipy.spatial.distance import cdist, euclidean
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

class GeneticAlgorithm(Algorithm):

    def __init__(self,
                 pop_size=None,
                 sampling=None,
                 selection=None,
                 crossover=None,
                 mutation=None,
                 survival=None,
                 n_offsprings=None,
                 eliminate_duplicates=DefaultDuplicateElimination(),
                 repair=None,
                 mating=None,
                 individual=Individual(),
                 min_infeas_pop_size=0,
                 **kwargs
                 ):

        super().__init__(**kwargs)

        """The population size used"""
        self.pop_size = pop_size

        """minimum number of individuals surviving despite being infeasible - by default disabled"""
        self.min_infeas_pop_size = min_infeas_pop_size

        """the survival for the genetic algorithm"""
        self.survival = survival

        """number of offsprings to generate through recombination"""
        self.n_offsprings = n_offsprings

        """if the number of offspring is not set - equal to population size"""
        if self.n_offsprings is None:
            self.n_offsprings = pop_size

        """The object to be used to represent an individual - either individual or derived class"""
        self.individual = individual

        """Set the duplicate detection class - a boolean value chooses the default duplicate detection"""
        if isinstance(eliminate_duplicates, bool):
            if eliminate_duplicates:
                self.eliminate_duplicates = DefaultDuplicateElimination()
            else:
                self.eliminate_duplicates = NoDuplicateElimination()
        else:
            self.eliminate_duplicates = eliminate_duplicates

        """Simply set the no repair object if it is None"""
        self.repair = repair if repair is not None else NoRepair()

        self.initialization = Initialization(sampling,
                                            #  individual=individual,
                                             repair=self.repair,
                                             eliminate_duplicates=self.eliminate_duplicates)

        if mating is None:
            mating = Mating(selection,
                            crossover,
                            mutation,
                            repair=self.repair,
                            eliminate_duplicates=self.eliminate_duplicates,
                            n_max_iterations=100)
        self.mating = mating

        """Other run specific data updated whenever solve is called - to share them in all algorithms"""
        self.n_gen = None
        self.pop = None
        self.off = None

        """Set the CO related parameters"""
        self.co_target = None
        self.co_start = False
        self.co_survived = []
        self.last_co_repair = 0

        """Set the DO related parameters"""
        self.do_state = [0, 0, 0]
        self.do_models = None
        self.do_survived = []
        self.last_do_learn = 0

        """initialize the termination archive as empty set."""
        self.mu_D = []
        self.D_t = []
        self.S_t = []
        self.termination_suggestion = [None,None,None]

    def _initialize(self):

        """Create the initial population"""
        pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        pop.set("n_gen", self.n_gen)

        """Then evaluate using the objective function"""
        self.evaluator.eval(self.problem, pop, algorithm=self)

        """That call is a dummy survival to set attributes that are necessary for the mating selection"""
        if self.survival:
            pop = self.survival.do(self.problem, pop, len(pop), algorithm=self,
                                   n_min_infeas_survive=self.min_infeas_pop_size)

        self.pop, self.off = pop, pop

        """Use 2D ref_dirs for 2-objective NSGA-II"""
        if self.problem.n_obj==2 or self.problem.n_obj==1:
            self.ref_dirs = get_reference_directions("das-dennis",2,n_partitions=self.pop_size-1)

        """For CO"""
        self.is_co_learn = True
        self.child = []
        X = np.array([None]*self.problem.n_var)
        F = np.array([None]*self.problem.n_obj)
        self.co_target = np.array([None]*len(self.ref_dirs))
        for i in range (len(self.co_target)):
            self.co_target[i] = [X,F]
        self.co_frequency = 1 #since it is anyways adapted on-the-fly
        self.do_frequency = 1 #since it is anyways adapted on-the-fly

        """For DO"""
        self.is_do_learn = True
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

    def _next(self):

        # print(self.co_frequency, self.do_frequency)
        if self.n_gen % 100 == 0: 
            print("Generation: " + str(self.n_gen))

        """Check if termination has been suggested. If yes, terminate"""
        check = [val for val in self.termination_suggestion if val is None]
        if len(check)==0:
            self.termination.force_termination = True
            return

        """Should we start CO?"""
        if self.is_co_learn and self.co_start:
            GeneticAlgorithm.update_co_target(self)
            if self.n_gen>self.collection+1 and self.n_gen-self.last_co_repair>=self.co_frequency:
                # print('CO learning now')
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
            Q = self.pop.get("F")
            parent_niches = self.pop.get("niche")
            parent_niches = np.unique(parent_niches)

        """Call the Learn function"""
        if self.is_do_learn and self.do_trigger2 and self.n_gen-self.last_do_learn>=self.do_frequency:
            # print('DO learning now')
            GeneticAlgorithm.do_learn(self)

        """do the mating using the current population"""
        self.off = self.mating.do(self.problem, self.pop, self.n_offsprings, algorithm=self)
        self.off.set("n_gen", self.n_gen)

        """if the mating could not generate any new offspring (duplicate elimination might make that happen)"""
        if len(self.off) == 0:
            self.termination.force_termination = True
            return

        # """if not the desired number of offspring could be created"""
        elif len(self.off) < self.n_offsprings:
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")

        """repair step:"""
        if self.n_gen>self.collection+1 and self.is_co_learn and self.n_gen-self.last_co_repair>=self.co_frequency and self.co_start:
            # print('CO repairing now')
            self.last_co_repair = self.n_gen
            GeneticAlgorithm.co_repair(self, lower_bound,upper_bound, RF_model)

        """Call the Repair function and replace offspring"""
        if self.is_do_learn and self.do_trigger2 and self.n_gen-self.last_do_learn>=self.do_frequency:
            temp = [1 for model in self.do_models if model is None]
            if sum(temp)==0:
                # print('DO repairing now')
                self.last_do_learn = self.n_gen
                new_solutions = GeneticAlgorithm.do_repair(self)
                sequence = np.arange(int(self.pop_size/2)) #DEXTER - offspring generated are already randomized - doesn't make sense to use another random input there
                for i in range(len(new_solutions)):
                    self.off[sequence[i]].X = np.array(new_solutions[i])

        """Evaluate the offspring"""
        self.evaluator.eval(self.problem, self.off, algorithm=self)
        if self.is_co_learn:
            self.child.append(self.off)

        """Merge the offsprings with the current population"""
        self.pop = Population.merge(self.pop, self.off)

        """The do survival selection"""
        if self.survival:
            self.pop = self.survival.do(self.problem, self.pop, self.pop_size, algorithm=self,
                                        n_min_infeas_survive=self.min_infeas_pop_size)

        """Update termination archives"""
        if self.do_trigger1==True:
            GeneticAlgorithm.update_termination_archives(self, Q)

        """co_frequency adaptation"""
        df1 = self.pop.get("F")
        df2 = self.off.get("F")[int(self.pop_size/2):]
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
        df2 = self.off.get("F")[:int(self.pop_size/2)]
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

    def _finalize(self):
        pass

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
        for row in self.callback.data["archive"][-(self.collection+1)]:
            obj_data.append(row.F)
            design_data.append(row.X)

        """Normalize and map the archive points to the targets"""
        nadir_point = self.pop[0].data['nadir_point']
        if nadir_point is None:
            nadir_point = np.max(self.pop.get("F"), axis=0)
        ideal_point = self.pop[0].data['ideal_point']
        if max(nadir_point) == math.inf:
            nadir_point = np.max(self.pop.get("F"), axis=0)
        obj_data = normalize(obj_data, x_min = ideal_point-1e-12, x_max = nadir_point)
        for i in range(len(design_data)):
            temp = load_function('calc_perpendicular_distance')([obj_data[i]],self.ref_dirs)[0]
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
        input_data = normalize(input_data,x_min=lower_bound,x_max=upper_bound)
        output_data = normalize(output_data,x_min=lower_bound,x_max=upper_bound)
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
            # print(iters)
            if iters<10:
                GeneticAlgorithm.co_boundary_repair(p,data,l,u,iters+1)
            else:
                # print(p,c,data)
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
        child = normalize(child,x_min=lower_bound,x_max=upper_bound)
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
            self.off[index].X = GeneticAlgorithm.co_boundary_repair(original_child,self.off[index].X,self.problem.xl,self.problem.xu,0)
            # self.off[index].X = GeneticAlgorithm.boundary_repair(original_child, self.off[index].X, self.problem)

    def update_co_target(self):
        """update target-dataset: [X,F]*pop_size"""
        X = [self.pop[i].X for i in range (self.pop_size)]
        F = [self.pop[i].F for i in range (self.pop_size)]

        """target normalization as per current pop"""
        nadir_point = self.pop[0].data['nadir_point']
        if nadir_point is None:
            nadir_point = np.max(self.pop.get("F"), axis=0)
        ideal_point = self.pop[0].data['ideal_point']
        if max(nadir_point) == math.inf:
            nadir_point = np.max(self.pop.get("F"), axis=0)
        F_norm = normalize(F, x_min = ideal_point-1e-12, x_max = nadir_point)
        assoc_index = []
        assoc_value = []
        for obj in F_norm:
            temp = load_function('calc_perpendicular_distance')([obj],self.ref_dirs)[0]
            assoc_index.append(np.argmin(temp))
            assoc_value.append(min(temp))

        """check if a population member can/should replace a target"""
        for i in range(self.pop_size):
            index = assoc_index[i]
            if self.co_target[index][0][0] is None:
                self.co_target[index][0] = X[i][:]
                self.co_target[index][1] = F[i][:]
            else:
                target_F = normalize(self.co_target[index][1], x_min = ideal_point-1e-12, x_max = nadir_point)
                if sum(target_F > F_norm[i]) >= 1 and sum(target_F >= F_norm[i]) == self.problem.n_obj:
                    self.co_target[index][0] = X[i][:]
                    self.co_target[index][1] = F[i][:]
                elif sum(target_F < F_norm[i]) >= 1 and sum(target_F > F_norm[i]) >= 1:
                    target_perp = load_function('calc_perpendicular_distance')([target_F],[self.ref_dirs[index]])[0][0]
                    if assoc_value[i] < target_perp:
                        self.co_target[index][0] = X[i][:]
                        self.co_target[index][1] = F[i][:]

    def do_predict(X, regr, problem, eta, category, niche_radius):
        X = normalize(X, x_min=problem.xl, x_max=problem.xu)
        X_ = regr.predict([X])[0]
        X_ = denormalize(X_, x_min=problem.xl, x_max=problem.xu)
        if category =='pit':
            eta = (-0.5+np.random.random())+eta
        else:
            eta = (np.random.random())*eta*(2**0.5)/niche_radius
        X_ = X + eta*(X_)
        X_ = GeneticAlgorithm.boundary_repair(X, X_, problem)
        # X_ = GeneticAlgorithm.co_boundary_repair(X, X_, problem.xl, problem.xu, 0)
        return X_

    def do_learn(self): 
        
        """Extract the data, normalize objectives"""
        # X = self.pop.get("X")
        # F = self.pop.get("F")
        X = np.array([ind.X for ind in self.pop if ind.data["rank"]==0]) 
        F = np.array([ind.F for ind in self.pop if ind.data["rank"]==0]) 
        nadir_point = self.pop[0].data['nadir_point']
        if nadir_point is None:
            nadir_point = np.max(self.pop.get("F"), axis=0)
        ideal_point = self.pop[0].data['ideal_point']
        if max(nadir_point) == math.inf:
            nadir_point = np.max(self.pop.get("F"), axis=0)
        F_norm = normalize(F, x_min=ideal_point-1e-12, x_max=nadir_point)
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
                input_data = normalize(input_data, x_min=self.problem.xl, x_max=self.problem.xu)
                output_data = normalize(output_data, x_min=self.problem.xl, x_max=self.problem.xu)
                self.do_models[i] = KNeighborsRegressor(n_neighbors=min(len(input_data), max(self.problem.n_obj, self.problem.n_var)))
                # self.do_models[i] = RandomForestRegressor(max_features=self.problem.n_var, min_samples_split=2, random_state=1, n_estimators=int(self.pop_size))
                self.do_models[i].fit(input_data, output_data)

    def do_repair(self):
        """extract the data, normalize objectives"""
        X = self.pop.get("X")
        F = self.pop.get("F") 
        nadir_point = self.pop[0].data['nadir_point']
        if nadir_point is None:
            nadir_point = np.max(self.pop.get("F"), axis=0)
        ideal_point = self.pop[0].data['ideal_point']
        if max(nadir_point) == math.inf:
            nadir_point = np.max(self.pop.get("F"), axis=0)
        F_norm = normalize(F, x_min=ideal_point-1e-12, x_max=nadir_point)
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
                dist = load_function('calc_perpendicular_distance')(F_norm, [pit])
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
                    temp = GeneticAlgorithm.do_predict(starting_point, self.do_models[index], self.problem, progress_length, 'pit', self.niche_radius)
                else:
                    temp = GeneticAlgorithm.do_predict(starting_point, self.do_models[index-self.problem.n_obj], self.problem, -1*progress_length, 'pit', self.niche_radius)
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
                    temp = GeneticAlgorithm.do_predict(starting_point, self.do_models[index_model], self.problem, 1, 'boundary', self.niche_radius)
                else:
                    temp = GeneticAlgorithm.do_predict(starting_point, self.do_models[index_model], self.problem, 1, 'boundary', self.niche_radius)
                new_solutions.append(temp)

        for i in range(len(new_solutions)):
            for j in range(len(new_solutions[i])):
                if math.isnan(new_solutions[i][j]):
                    print('NaN in DO-offspring')
        
        return new_solutions

    def update_termination_archives(self,Q):

        #gather data
        P = self.pop.get("F")
        nadir_point = self.pop[0].data['nadir_point']
        if nadir_point is None:
            nadir_point = np.max(self.pop.get("F"), axis=0)
        ideal_point = self.pop[0].data['ideal_point']
        if max(nadir_point) == math.inf:
            nadir_point = np.max(self.pop.get("F"), axis=0)

        P_norm = normalize(P, x_min=ideal_point-1e-12, x_max=nadir_point)
        Q_norm = normalize(Q, x_min=ideal_point-1e-12, x_max=nadir_point)

        # association with ref-vectors
        assoc_P = np.array([ind.data['niche'] for ind in self.pop])
        dist_matrix = load_function('calc_perpendicular_distance')(Q_norm, self.ref_dirs)
        assoc_Q = np.array([np.argmin(row) for row in dist_matrix])

        # termination-1 loop for each reference-vector (only for convergence)
        mu_D = 0
        count = 0
        for i in range (len(self.ref_dirs)):
            cluster_P = np.where(assoc_P==i)[0]
            cluster_Q = np.where(assoc_Q==i)[0]
            if len(cluster_P)>0 and len(cluster_Q)>0:
                p = np.mean(P_norm[cluster_P], axis=0)
                q = np.mean(Q_norm[cluster_Q], axis=0)
                a = np.matmul(self.ref_dirs[i],p)
                b = np.matmul(self.ref_dirs[i],q)
                D = abs(a - b)
                if max(a, b)!=0:
                    D = D/max(a, b)
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
        # print(mu_D)
        #update the archives
        self.mu_D.append(mu_D)
        self.D_t.append(np.mean(self.mu_D))
        self.S_t.append(np.std(self.mu_D))

    def check_for_termination(self, n_p, ns):

        to_terminate = False
        if len(self.mu_D)>ns:
            D_t = self.D_t[-ns:]
            S_t = self.S_t[-ns:]
            D_t = [round(val,n_p) for val in D_t]
            S_t = [round(val,n_p) for val in S_t]
            # print(len(np.unique(D_t)), D_t)
            if len(np.unique(D_t))==1 and len(np.unique(S_t))==1:
                to_terminate = True
            return to_terminate

    # def do_adaptation(self, new_solutions, sequence, Q, parent_niches, parent_ideal_point):
    #     gap_repairs = int(self.pop_size*self.do_repair_fraction[0])
    #     bound_repairs = int(self.pop_size*self.do_repair_fraction[1])
    #     total_repairs = gap_repairs + bound_repairs
    #     if len(new_solutions)==bound_repairs:
    #         gap_solutions = []
    #         bound_solutions = sequence[:bound_repairs]
    #     elif len(new_solutions)==total_repairs:
    #         gap_solutions = sequence[:gap_repairs]
    #         bound_solutions = sequence[gap_repairs:total_repairs]
    #     elif len(new_solutions)==gap_repairs:
    #         gap_solutions = sequence[:gap_repairs]
    #         bound_solutions = []
    #     nadir_point = self.pop[0].data['nadir_point']
    #     ideal_point = self.pop[0].data['ideal_point']
    #     Q = normalize(Q, x_min=ideal_point, x_max=nadir_point)
    #     pop = self.pop.get("X")

    #     """Analyze the gap survival"""
    #     if len(gap_solutions)>0:
    #         X = self.off.get("X")[gap_solutions]
    #         gap_index = []
    #         for i in range(len(X)):
    #             dist = [sum(np.abs(ind-X[i])) for ind in pop]
    #             if min(dist)==0:
    #                 gap_index.append(gap_solutions[i])
    #         # gap_delta = len(gap_index)/len(gap_solutions)
    #         gap_delta = []
    #         if len(gap_index)>0:
    #             F = self.off.get("F")[gap_index]
    #             F = normalize(F, x_min=ideal_point, x_max=nadir_point)
    #             for f in F:
    #                 dist = min(cdist([f], Q)[0])
    #                 gap_delta.append(dist)
    #             # gap_delta = np.mean(gap_delta)
    #             gap_delta = sum(gap_delta)/len(gap_solutions)
    #         else:
    #             gap_delta = 0
    #     else:
    #         gap_delta = 0

    #     """Analyze the boundary survival"""
    #     X = self.off.get("X")[bound_solutions]
    #     bound_index = []
    #     for i in range(len(X)):
    #         dist = [sum(np.abs(ind-X[i])) for ind in pop]
    #         if min(dist)==0:
    #             bound_index.append(bound_solutions[i])
    #     # bound_delta = len(bound_index)/len(bound_solutions)
    #     bound_delta = []
    #     if len(bound_index)>0:
    #         F = self.off.get("F")[bound_index]
    #         F = normalize(F, x_min=ideal_point, x_max=nadir_point)
    #         for f in F:
    #             dist = min(cdist([f], Q)[0])
    #             bound_delta.append(dist)
    #         # bound_delta = np.mean(bound_delta)
    #         bound_delta = sum(bound_delta)/len(bound_solutions)
    #     else:
    #         bound_delta = 0

    #     """Analyze the crossover-mutation survival"""
    #     repair_solutions = [val for val in gap_solutions]
    #     repair_solutions.extend(bound_solutions)
    #     crossover_solutions = np.array([i for i in range(len(self.off)) if i not in repair_solutions])
    #     # crossover_solutions = crossover_solutions.astype(int)
    #     X = self.off.get("X")[crossover_solutions]
    #     crossover_index = []
    #     for i in range(len(X)):
    #         dist = [sum(np.abs(ind-X[i])) for ind in pop]
    #         if min(dist)==0:
    #             crossover_index.append(crossover_solutions[i])
    #     # crossover_delta = len(crossover_index)/len(crossover_solutions)
    #     crossover_delta = []
    #     if len(crossover_index)>0:
    #         F = self.off.get("F")[crossover_index]
    #         F = normalize(F, x_min=ideal_point, x_max=nadir_point)
    #         for f in F:
    #             dist = min(cdist([f], Q)[0])
    #             crossover_delta.append(dist)
    #         # crossover_delta = np.mean(crossover_delta)
    #         crossover_delta = sum(crossover_delta)/len(crossover_solutions)
    #     else:
    #         crossover_delta = 0

    #     """Assistance to Rg and Rb to die down eventually"""
    #     reduce_Rg  = False
    #     reduce_Rb  = False
    #     survived_pop_niches = self.pop.get("niche")
    #     survived_pop_niches = np.unique(survived_pop_niches)
    #     survived_ideal_point = np.min(self.pop.get("F"), axis=0)
    #     if len(parent_niches)==len(survived_pop_niches):
    #         if sum(parent_niches - survived_pop_niches)==0:
    #             reduce_Rg = True
    #     if sum(parent_ideal_point-survived_ideal_point)==0:
    #         reduce_Rb = True

    #     # print([gap_delta, bound_delta, crossover_delta])
    #     self.survival_history = np.array([gap_delta, bound_delta, crossover_delta])
    #     if self.survival_history[0] < self.survival_history[2] or reduce_Rg:
    #         self.do_repair_fraction[0] -= self.adaptation
    #     else:
    #         self.do_repair_fraction[0] += self.adaptation
    #     if self.survival_history[1] < self.survival_history[2] or reduce_Rb:
    #         self.do_repair_fraction[1] -= self.adaptation
    #     else:
    #         self.do_repair_fraction[1] += self.adaptation

    #     # if reduce_Rg:
    #     #     self.do_repair_fraction[0] -= self.adaptation
    #     # if reduce_Rb:
    #     #     self.do_repair_fraction[1] -= self.adaptation

    #     if self.do_repair_fraction[0] < self.min_repair[0]:
    #         self.do_repair_fraction[0] = self.min_repair[0]
    #     elif self.do_repair_fraction[0] > self.max_repair[0]:
    #         self.do_repair_fraction[0] = self.max_repair[0]
    #     if self.do_repair_fraction[1] < self.min_repair[1]:
    #         self.do_repair_fraction[1] = self.min_repair[1]
    #     elif self.do_repair_fraction[1] > self.max_repair[1]:
    #         self.do_repair_fraction[1] = self.max_repair[1]
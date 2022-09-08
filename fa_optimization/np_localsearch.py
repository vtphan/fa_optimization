#------------------------------------------------------------------------------
#
# Author: Vinhthuy Phan, 2021
#
#------------------------------------------------------------------------------
import pandas
import numpy
import joblib
import os, datetime
from .largest_k import LargestK
import datetime

class NPLocalSearch:
    def __init__(self, s0, model, opt_attributes, opt_weights, k_best, output_file='optimizer_results.csv'):
        self.s0 = s0
        self.solution = s0.copy()
        self.model = model
        self.model.apply_need_merit_buckets(self.solution)
        self.opt_attributes = opt_attributes
        self.opt_weights = numpy.array(opt_weights)
        self.normalizer = self.model.value[opt_attributes].copy()
        self.history = []
        self.k_best = k_best
        self.best_solutions = None
        self.best_solution = None
        self.best_score = None
        self.runtime = 0
        self.output_file = output_file

    #--------------------------------------------------------------------------
    def reset(self):
        self.solution = self.s0.copy()
        self.model.init_stats()
        self.model.apply_need_merit_buckets(self.solution)

    #--------------------------------------------------------------------------
    def determine_which_to_optimize(self, m_amounts, n_amounts, prob=0.5):
        if n_amounts==[] and m_amounts!=[]:
            return 'merit'
        if n_amounts!=[] and m_amounts==[]:
            return 'need'
        if n_amounts!=[] and m_amounts!=[]:
            r = numpy.random.rand()
            if r<=prob:
                return 'merit'
            else:
                return 'need'
        return 'none'

    #--------------------------------------------------------------------------
    def stochastic_hill_climbing(self, m_amounts=[], n_amounts=[], add_prob=0.5, merit_prob=0.5, iterations=50,
        max_investment = None):
        starting_time = datetime.datetime.now()
        
        self.best_solutions = LargestK(self.k_best)
        self.shc_worst_value = None
        self.shc_best_value = None
        self.hc_iterations = iterations
        forward_moves = 0
        no_progress, no_progress_steps = False, 0

        for i in range(iterations):
            wto = self.determine_which_to_optimize(m_amounts, n_amounts, merit_prob)
            if wto == 'merit':
                buckets = self.solution.merit_buckets
                apply_buckets = self.model.apply_merit_buckets
                amounts = m_amounts
            elif wto == 'need':
                buckets = self.solution.need_buckets
                apply_buckets = self.model.apply_need_buckets
                amounts = n_amounts
            else:
                break

            r = numpy.random.rand()
            if r <= add_prob:
                amount, idx = buckets.random_add(amounts)
                op = 'add'
            else:
                amount, idx = buckets.random_remove(amounts)
                op = 'remove'
            if idx>=0:
                self.model.save_value()
                apply_buckets(self.solution)
                previous_score = self.previous_score()
                score = self.score()
                
                if self.shc_worst_value is None or self.shc_worst_value > score:
                    self.shc_worst_value = score
                if self.shc_best_value is None or self.shc_best_value < score:
                    self.shc_best_value = score
                    
                constraints_satisfied = False
                if max_investment is None or self.model.value.amount_mn <= max_investment:
                    constraints_satisfied = True 

                if score > previous_score and constraints_satisfied:
                    forward_moves += 1
                    no_progress, no_progress_steps = False, 0
                    print('\ti={}, {}, {} {}, score increased to {}.'.format(i,wto,op,amount,round(score,6)))
                else:
                    no_progress, no_progress_steps = True, no_progress_steps+1
                    buckets.undo(op, amount, idx)
                    self.model.value = self.model.saved_value.copy()
                    #
                    # Skip applying old solution to buckets
                    # because new solution will be applied anyway (line 57)
                    #
                    # apply_buckets(self.solution)
                    #
                if constraints_satisfied:
                    self.best_solutions.add(score, self.solution, copy=True)
            else:
                if wto == 'merit':
                    # m_amounts=[]
                    print('Unable to make improvement on merit buckets at iteration {}'.format(i))
                elif wto == 'need':
                    # n_amounts = []
                    print('Unable to make improvement on need buckets at iteration {}'.format(i))
                else:
                    print('Unable to make improvement on iteration {}'.format(i))
                    continue

        self.best_score, self.best_solution = self.best_solutions.largest()
        if self.best_solution is not None:
            self.model.apply_need_merit_buckets(self.best_solution)
        self.shc_best_solution = self.best_solution.copy()
        self.archive('stochastic hill climbing',
            m_amounts=m_amounts,
            n_amounts=n_amounts,
            add_prob=add_prob,
            merit_prob=merit_prob,
            iterations=iterations,
        )
        # print(self.best_solution)
        self.runtime = datetime.datetime.now() - starting_time
        self.append_results('stochastic_hc')
        return forward_moves / iterations, no_progress_steps / iterations

    #--------------------------------------------------------------------------
    def simulated_quenching(self, 
                            m_amounts=[], 
                            n_amounts=[], 
                            add_prob=0.5, 
                            merit_prob=0.5, 
                            temperature_duration=20,
                            temperature_decrement=0.95,
                            annealing_iterations=50,
                            start_from_s0=False,
                            max_investment=None):
        starting_time = datetime.datetime.now()
        print('Optimizing for {} with weights {}'.format(self.opt_attributes, self.opt_weights))
        print(self.solution)
        print('Stochastic hill climb')
        self.stochastic_hill_climbing(m_amounts, n_amounts, iterations=annealing_iterations)
        print()
        print()
        print('Simulated quenching')
        if start_from_s0:
            self.solution = self.s0.copy()
        T = (self.shc_worst_value-self.shc_best_value)/numpy.log(0.5)
        self.best_solutions = LargestK(self.k_best)
        self.sq_iterations = annealing_iterations
        self.sq_temp_duration = temperature_duration
        for j in range(annealing_iterations):
            T = T*temperature_decrement
            forward_moves, backward_moves = 0, 0
            for i in range(temperature_duration):
                wto = self.determine_which_to_optimize(m_amounts, n_amounts, merit_prob)
                if wto == 'merit':
                    buckets = self.solution.merit_buckets
                    apply_buckets = self.model.apply_merit_buckets
                    amounts = m_amounts
                elif wto == 'need':
                    buckets = self.solution.need_buckets
                    apply_buckets = self.model.apply_need_buckets
                    amounts = n_amounts
                else:
                    break

                r = numpy.random.rand()
                if r <= add_prob:
                    amount, idx = buckets.random_add(amounts)
                    op = 'add'
                else:
                    amount, idx = buckets.random_remove(amounts)
                    op = 'remove'
                if idx>=0:
                    self.model.save_value()
                    apply_buckets(self.solution)
                    previous_score = self.previous_score()
                    score = self.score()

                    constraints_satisfied = False
                    if max_investment is None or self.model.value.amount_mn <= max_investment:
                        constraints_satisfied = True 

                    if score > previous_score and constraints_satisfied:
                        print('\ti={}, {}, {} {}, score increased to {}.'.format(
                            i,wto,op,amount,round(score,6)))
                        forward_moves += 1
                    else:
                        transition_prob = numpy.e**((score-previous_score)/T)
                        if numpy.random.rand() <= transition_prob and constraints_satisfied:
                            backward_moves += 1
                            print('\ti={}, {}, {} {}, backward move with score decreased to {}.'.format(
                                i,wto,op,amount,round(score,6)))
                        else:
                            buckets.undo(op, amount, idx)
                            self.model.value = self.model.saved_value.copy()

                    if constraints_satisfied:
                        self.best_solutions.add(score, self.solution, copy=True)
                else:
                    if wto == 'merit':
                        # m_amounts=[]
                        print('\tUnable to make improvement on merit buckets at iteration {}'.format(i))
                    elif wto == 'need':
                        # n_amounts = []
                        print('\tUnable to make improvement on need buckets at iteration {}'.format(i))
                    else:
                        print('\tUnable to make improvement on iteration {}'.format(i))
                        continue
                        
            print('j={}, T={}, forward%={}, backward%={}'.format(
                j+1,
                round(T,8),
                round(forward_moves/temperature_duration,4),
                round(backward_moves/temperature_duration,4)))


        self.best_score, self.best_solution = self.best_solutions.largest()
        if self.best_solution is not None:
            self.model.apply_need_merit_buckets(self.best_solution)

        self.archive('simulated quenching',
            m_amounts=m_amounts,
            n_amounts=n_amounts,
            add_prob=add_prob,
            merit_prob=merit_prob,
            temperature_duration=temperature_duration,
            temperature_decrement=temperature_decrement,
            annealing_iterations=annealing_iterations,
        )
        self.runtime = datetime.datetime.now() - starting_time
        self.append_results('simulated_quenching')


    #--------------------------------------------------------------------------
    def _score(self, value):
        v = value[self.opt_attributes]
        if 'amount_mn' in v:
            v.amount_mn = 1.0 / v.amount_mn
        #
        # this could be better:  (v-self.normalizer) /  abs(self.normalizer)
        #
        return (v/self.normalizer).dot(self.opt_weights)



    def score(self):
        return self._score(self.model.value)

    def previous_score(self):
        return self._score(self.model.saved_value)

    #--------------------------------------------------------------------------
    def archive(self, message, **argv):
        now = datetime.datetime.now().strftime('%Y.%m.%d %H:%M:%S')
        record = (message, now, argv, str(self.best_solution))
        self.history.append(record)
        print('This run has been archived at {}.'.format(now))

    #--------------------------------------------------------------------------
    def append_results(self, optimizer):
        if self.best_solution is None:
            print('Best solution does not exist. Nothing done.')
            return
        
        s = self.best_solution 
        score = self.best_score
        
        full_stats = self.model.create_full_stats()
        best_value = self.model.compute_value(full_stats, self.model.y_prob)
        baseline = self.model.baseline
        iterations = ''
        if optimizer=='stochastic_hc':
            iterations = str(self.hc_iterations)
        elif optimizer=='simulated_quenching':
            iterations = '{},{}'.format(self.sq_iterations, self.sq_temp_duration)
        elif optimizer=='simulated_annealing':
            iterations = '{},{}'.format(self.sa_iterations, self.sa_temp_duration)

        result = dict(
            optimizer = [optimizer],
            opt_weights = [self.opt_weights],
            opt_attributes = [self.opt_attributes],
            iterations = [iterations],
            amount_merit = [100*(best_value.amount_merit/baseline.amount_merit - 1)],
            amount_merit_diff = [best_value.amount_merit-baseline.amount_merit],
            amount_need = [100*(best_value.amount_need/baseline.amount_need - 1)],
            amount_need_diff = [best_value.amount_need-baseline.amount_need],
            amount_mn = [100*(best_value.amount_mn/baseline.amount_mn - 1)],
            amount_mn_diff = [best_value.amount_mn-baseline.amount_mn],
            enrollment = [100*(best_value.enrollment/baseline.enrollment - 1)],
            revenue = [100*(best_value.revenue/baseline.revenue - 1)],
            revenue_diff = [best_value.revenue - baseline.revenue],
            achievement_index = [100*(best_value.achievement_index/baseline.achievement_index - 1)],
            roi = [best_value.roi - baseline.roi],
            affordability = [100*(best_value.affordability/baseline.affordability - 1)],
            accessibility = [100*(best_value.accessible/baseline.accessible - 1)],
            unmet_need = [best_value.affordable-baseline.affordable],
            achievement = [100*(best_value.achievement/baseline.achievement - 1)],
            merit_buckets = [s.merit_buckets.value],
            need_buckets = [s.need_buckets.value],
            score = [score],
            runtime = [self.runtime.seconds],
            now = [datetime.datetime.now()],
        )
        df = pandas.DataFrame(result)
        if not os.path.isfile(self.output_file):
            df.to_csv(self.output_file, index=False)
        else:
            df.to_csv(self.output_file, index=False, header=False, mode='a')
        return df
        
    #--------------------------------------------------------------------------
    def save_to(self, dir=None):
        if dir is None:
            print('Must provide a directory name to save optimizer.')
            return
        if not os.path.exists(dir):
            os.mkdir(dir)
        today = datetime.datetime.today()
        prefix = 'optimizer_{}_{}_{}_'.format(today.year, today.month, today.day)
        n = len([f for f in os.listdir(dir) if f.startswith(prefix)])
        filename = os.path.join(dir, prefix + str(n+1))
        joblib.dump(self, filename)
        print("Optimizer is saved to {}\nTo load it, use Optimizer.load('{}')".format(filename,filename))

    #--------------------------------------------------------------------------
    @classmethod
    def load(cls, filename):
        return joblib.load(filename)

    #--------------------------------------------------------------------------
    def help(self):
        print('''
Stochastic Hill Climbing parameters:
    m_amounts
        Merit amounts to be added/removed randomly in each step of hill climbing.
        If not provided, then only need buckets are optimized.
    n_amounts
        Need amounts to be added/removed randomly in each step of hill climbing.
        If not provided, then only merit buckets are optimized.
    add_prob
        Probability of adding an amount in each step. 
        Removing prob = 1-add_prob.
        Default value is 0.5.
    merit_prob
        Probability of optimizing merit buckets in each step. 
        Need_prob = 1-merit_prob.
        Default value is 0.5.
    iterations
        Number of iterations of hill climbing.
        Default value is 50.

Simulated Annealing parameters:
    m_amounts 
        Merit amounts to be added/removed randomly in each step of hill climbing.
        If not provided, then only need buckets are optimized.
    n_amounts 
        Need amounts to be added/removed randomly in each step of hill climbing.
        If not provided, then only merit buckets are optimized.
    add_prob
        Probability of adding an amount in each step. 
        Removing prob = 1-add_prob.
        Default value is 0.5.
    merit_prob
        Probability of optimizing merit buckets in each step. 
        Need_prob = 1-merit_prob.
        Default value is 0.5.
    temperature_duration
        Number of iterations each temperature lasts.
        Default value is 20.
    temperature_decrement
        Constant decrement fraction of the initial temperature (T0)
        Must be less than 1.
        Default value is 0.01. 
    annealing_iterations
        Number of iterations of simulated annealing.
        Default value is 50.
            ''')
        
    #--------------------------------------------------------------------------





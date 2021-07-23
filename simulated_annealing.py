from math import exp
from random import choices
from weightless_neural_network import WeighlessNetwok
import numpy as np


class SimulatedAnnealing():
    def __init__(self, data, steps, temperature, t_function, bounds = []):
        #self.bounds = [(3, 10), (3, 36), (1, 9), (3, 10), (3, 10)]
        self.bounds = [(3, 10), (3, 36), (1, 9), (3, 10)]
        self.temperature = temperature
        self.t_function = t_function
        self.steps = steps
        #self.initial_state = [3, 3, 1, 3, 3]
        self.initial_state = [3, 3, 1, 3]
        self.wn = WeighlessNetwok(data)
        self.state_evals = dict()

    def train_and_evaluate(self, candidate):
        candidate_eval = self.state_evals.get(tuple(candidate))
        if candidate_eval is not None:
            return candidate_eval
        pred = self.wn.train(candidate)
        eval = self.wn.eval(pred)
        self.state_evals[tuple(candidate)] = eval
        return eval

    def choose_list_candidates(self, curr_state):
        candidates = list()
        for arg in range(len(curr_state)):
            new_state_ahead = list(curr_state) 
            new_state_behind = list(curr_state)
            if curr_state[arg] == self.bounds[arg][0]:
                new_state_behind[arg] = self.bounds[arg][1]
            else:
                new_state_behind[arg] -= 1

            if curr_state[arg] == self.bounds[arg][1]:
                new_state_ahead[arg] = self.bounds[arg][0]
            else:
                new_state_ahead[arg] += 1
            candidates.append(new_state_ahead)
            candidates.append(new_state_behind)

        candidate_position = int(np.random.uniform()*len(candidates))
        return candidates[candidate_position]
        

    def choose_next_state(self, zipped_candidates):
        filtered_candidates = list(filter(lambda x: x[1] >= 0, zipped_candidates))
        population = list(range(0, len(filtered_candidates)))
        weight = 1/len(filtered_candidates)
        state_position = choices(population, [weight]*len(filtered_candidates))

        print('N ', filtered_candidates[state_position[0]])
        return filtered_candidates[state_position[0]]

    def choose_next_state_metropolis(self, zipped_candidates, curr_state, curr_state_eval):
        population = list(range(0, len(zipped_candidates) + 1))
        weights = list(list(zip(*zipped_candidates))[1])
        total_weight = 1 - sum(weights)
        weights.append(total_weight if (total_weight > 0) else 0)
        print('M weights ', weights)
        state_position = choices(population, weights)
        zipped_candidates.append((curr_state, curr_state_eval))

        print('M ', zipped_candidates[state_position[0]])
        return zipped_candidates[state_position[0]]

    def choose_next_state_metropolis_1(self, metropolis, curr_state, candidate_state):
        if np.random.uniform() <= metropolis:
            return candidate_state
        return curr_state


    def metropolis(self, curr_state_eval, candidate_eval, curr_temp):
        diff = candidate_eval-curr_state_eval
        return 1/(2*len(self.bounds))*min(exp(diff/curr_temp), 1)

    def sort_state_candidate(self, curr_state):
        new_state_candidate = list(curr_state)
        parameter = int(np.random.uniform()*4)
        parameter_bounds = self.bounds[parameter]
        param_value = np.random.uniform(low=parameter_bounds[0], high=parameter_bounds[1]+1)
        new_state_candidate[parameter] = param_value
        return new_state_candidate


    def simulate(self):
        best = self.initial_state
        best_eval = self.train_and_evaluate(best)
        curr_state, curr_state_eval = best, best_eval
        curr_temp = self.temperature

        for i in range(self.steps):
            candidate = self.choose_list_candidates(curr_state)
            #candidate = self.sort_state_candidate(curr_state)

            candidate_eval = self.train_and_evaluate(candidate)

            if (candidate_eval > curr_state_eval):
                best, best_eval = candidate, candidate_eval 
                curr_state, curr_state_eval = candidate, candidate_eval        
            else:
                metropolis = self.metropolis(curr_state_eval, candidate_eval, curr_temp)
                curr_state = self.choose_next_state_metropolis_1(metropolis, 
                                                                curr_state, 
                                                                candidate)
            curr_temp = self.t_function(i, self.temperature)
        return [best, best_eval]
            
import numpy as np
from math import exp
from random import choices
from weightless_neural_network import WeighlessNetwok


class SimulatedAnnealing():
    def __init__(self, steps, temperature, t_function, data, bounds = []):
        self.bounds = [(3, 10), (3, 36), (0.1, 0.9), (3, 10), (3, 10)]
        self.temperature = temperature
        self.t_function = t_function
        self.steps = steps
        self.initial_state = list()
        self.wn = WeighlessNetwok(data)

    def objective(self, candidates):
        evals = list()
        for candidate in candidates:
            pred = self.wn.train(candidate)
            evals.append(self.wn.eval(pred))
        return evals

    def list_candidates(self, curr_state):
        candidates = list()
        for arg in range(len(curr_state)):
            new_state_ahead, new_state_behind = curr_state, curr_state
            if curr_state[arg] == self.bounds[arg][0]:
                new_state_behind[arg] = self.bounds[arg][1]
            else:
                new_state_ahead[arg] += 1

            if curr_state[arg] == self.bounds[arg][1]:
                new_state_ahead[arg] = self.bounds[arg][0]
            else:
                new_state_behind[arg] -= 1                    
            candidates.append(new_state_ahead)
            candidates.append(new_state_behind)
            
        return candidates

    def metropolis(self, zipped_diffs, curr_temp):
        m_list = list()
        for candidate, _, diff in zipped_diffs:
            if diff < 0:
                 m_list.append((candidate, (1/10)*min(exp(-diff/curr_temp), 1)))
        return m_list

    def choose_next_state(self, zipped_candidates):
        filtered_candidates = list(filter(lambda x: x[1] >= 0, zipped_candidates))
        population = list(range(0, len(filtered_candidates)))
        weight = 1/len(filtered_candidates)
        state_position = choices(population, [weight]*len(filtered_candidates))

        return filtered_candidates[state_position]

    def choose_next_state_metropolis(self, zipped_candidates, curr_state, curr_state_eval):
        population = list(range(0, len(zipped_candidates) + 1))
        weights = list(list(zip(*zipped_candidates))[1])
        total_weight = 1 - sum(weights)
        weights.append(total_weight if (total_weight > 0) else 0)
        state_position = choices(population, weights)
        zipped_candidates.append((curr_state, curr_state_eval, 0))

        return zipped_candidates[state_position]

    def simulate(self):
        best = self.initial_state
        best_eval = self.objective(best)
        curr_state, curr_state_eval = best, best_eval
        curr_temp = self.temperature

        for i in range(self.steps):
            candidates = self.list_candidates(curr_state)
            candidates_eval = self.objective(candidates)
            diffs = [candidate_eval - curr_state_eval for candidate_eval in candidates_eval]
            zipped_diffs = zip(candidates, candidates_eval, diffs)
            if np.where(diffs < 0 ):
                metropolis = self.metropolis(zipped_diffs, curr_temp)
                candidate, candidate_eval, _ = self.choose_next_state_metropolis(metropolis, 
                                                                                curr_state, 
                                                                                curr_state_eval)
            else:
                candidate, candidate_eval, _ = self.choose_next_state(zipped_diffs)
                best, best_eval = candidate, candidate_eval
            curr_state, curr_state_eval = candidate, candidate_eval
            curr_temp = self.t_function(i, self.temperature)
        return [best, best_eval]
            
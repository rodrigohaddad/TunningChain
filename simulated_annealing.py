import numpy as np
from math import exp

class SimulatedAnnealing():
    def __init__(self, steps, temperature, t_function):
        self.temperature = temperature
        self.t_function = t_function
        self.steps = steps
        self.initial_state = list()

    def objective(self, candidates):
        for candidate in candidates:
            pass

    def list_candidates(self, curr_state):
        pass

    def metropolis(self, zipped_diffs, curr_temp):
        m_list = list()
        for candidate, diff in zipped_diffs:
            if diff < 0:
                 m_list.append((candidate, (1)*exp(-diff/curr_temp)))
        return m_list
            

    def choose_next_state(self, zipped_candidates):
        pass

    def choose_next_state_metropolis(self):
        pass

    def simulate(self):
        best = self.initial_state
        best_eval = self.objective()
        scores = list()
        curr_state, curr_state_eval = best, best_eval
        curr_temp = self.temperature

        for i in range(self.steps):
            candidates = self.list_candidates(curr_state)
            candidates_eval = self.objective(candidates)
            diffs = [candidate_eval - curr_state_eval for candidate_eval in candidates_eval]
            if np.where(diffs < 0 ):
                zipped_diffs = zip(candidates, diffs)
                metropolis = self.metropolis(zipped_diffs, curr_temp)
                candidate, candidate_eval = self.choose_next_state_metropolis(metropolis)
            else:
                zipped_candidates = zip(candidates, candidates_eval)
                candidate, candidate_eval = self.choose_next_state(zipped_candidates)
                best, best_eval = candidate, candidate_eval
            curr_state, curr_state_eval = candidate, candidate_eval
            curr_temp = self.t_function(i, self.temperature)
        return [best, best_eval]
            
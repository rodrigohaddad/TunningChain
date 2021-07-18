from math import exp
from random import choices
from weightless_neural_network import WeighlessNetwok
import concurrent.futures


class SimulatedAnnealing():
    def __init__(self, data, steps, temperature, t_function, bounds = []):
        #self.bounds = [(3, 10), (3, 36), (1, 9), (3, 10), (3, 10)]
        self.bounds = [(3, 10), (3, 36), (1, 9), (3, 10),]
        self.temperature = temperature
        self.t_function = t_function
        self.steps = steps
        #self.initial_state = [3, 3, 1, 3, 3]
        self.initial_state = [3, 3, 1, 3]
        self.wn = WeighlessNetwok(data)

    def train_and_evaluate(self, candidate):
        pred = self.wn.train(candidate)
        return self.wn.eval(pred)

    def objective(self, candidates):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures_process = [
                executor.submit(
                    self.train_and_evaluate, candidate
                ) for candidate in candidates
            ]
            executor.shutdown(wait=True)
            futures, _ = concurrent.futures.wait(futures_process)
            evals = list()
            for future in futures:
                try:
                    evals.append(future.result())
                except Exception as exc:
                    print(f"exe: {exc}")
        print('evals ', evals)
        return evals

    def list_candidates(self, curr_state):
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
        
        return candidates

    def metropolis(self, zipped_diffs, curr_temp):
        m_list = list()
        for candidate, _, diff in zipped_diffs:
            if diff < 0:
                m_list.append((candidate, (1/(2*len(self.bounds)))*min(exp(diff/curr_temp), 1)))
        return m_list

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

    def simulate(self):
        best = self.initial_state
        best_eval = self.objective([best])
        curr_state, curr_state_eval = best, best_eval
        curr_temp = self.temperature

        for i in range(self.steps):
            candidates = self.list_candidates(curr_state)
            candidates_eval = self.objective(candidates)
            diffs = [candidate_eval - curr_state_eval for candidate_eval in candidates_eval]
            zipped_diffs = zip(candidates, candidates_eval, diffs)
            if any(y > 0 for y in diffs):
                candidate, candidate_eval, _ = self.choose_next_state(zipped_diffs)
                best, best_eval = candidate, candidate_eval         
            else:
                metropolis = self.metropolis(zipped_diffs, curr_temp)
                candidate, candidate_eval = self.choose_next_state_metropolis(metropolis, 
                                                                                curr_state, 
                                                                                curr_state_eval)
            curr_state, curr_state_eval = candidate, candidate_eval
            curr_temp = self.t_function(i, self.temperature)
        return [best, best_eval]
            
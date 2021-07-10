from math import exp

class SimulatedAnnealing():
    def __init__(self, steps):
        self.steps = steps
        self.initial_state = list()

    def objective(self):
        pass

    def simulate(self):
        best = self.initial_state
        best_eval = self.objective()
        scores = list()

        for i in range(self.steps):
            # take a step


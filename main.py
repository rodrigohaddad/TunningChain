from simulated_annealing import SimulatedAnnealing
import numpy as np

def load(f):
        return np.load(f)['arr_0']

def main():
    def t_function_1():
        return 1
    def t_function_2():
        return 2
    def t_function_3():
        return 3
    params = [{'steps': 10, 'temperature': 10, 't_function': t_function_1},
            {'steps': 10, 'temperature': 10, 't_function': t_function_2},
            {'steps': 10, 'temperature': 10, 't_function': t_function_3}]

    data = {'x_train': load('images/kmnist-train-imgs.npz'),
        'x_test': load('images/kmnist-test-imgs.npz'),
        'y_train': [str(i) for i in load('images/kmnist-train-labels.npz')],
        'y_test': [str(i) for i in load('images/kmnist-test-labels.npz')]}

    annealing_results = list()
    for kwargs in params:
        annealing_results.append(SimulatedAnnealing(**kwargs, data))

    return annealing_results

if __name__ == '__main__':
    print(main())

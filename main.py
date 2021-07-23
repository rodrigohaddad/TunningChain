from simulated_annealing import SimulatedAnnealing
import numpy as np

def load(f):
        return np.load(f)['arr_0']

def main():
    def t_function_1(t, t0, beta=0.65):
        return t0*beta**t
    def t_function_2(t, t0, beta=0.65):
        return (t0-beta*t)
    params = [{'steps': 50, 'temperature': 50, 't_function': t_function_1},
            #{'steps': 10, 'temperature': 10, 't_function': t_function_2}
            ]

    data = {'x_train': load('images/kmnist-train-imgs.npz'),
        'x_test': load('images/kmnist-test-imgs.npz'),
        'y_train': [str(i) for i in load('images/kmnist-train-labels.npz')],
        'y_test': [str(i) for i in load('images/kmnist-test-labels.npz')]}

    data_2 = {'x_train': data['x_train'][:300],
        'x_test': data['x_test'][:50],
        'y_train': data['y_train'][:300],
        'y_test': data['y_test'][:50]}

    annealing_results = list()
    for kwargs in params:
        print(kwargs)
        sa = SimulatedAnnealing(data=data_2, **kwargs)
        annealing_results.append(sa.simulate())

    return annealing_results

if __name__ == '__main__':
    print(main())

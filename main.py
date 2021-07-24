from math import log
from simulated_annealing import SimulatedAnnealing
import numpy as np
import json
from sklearn.model_selection import train_test_split

def load(f):
        return np.load(f)['arr_0']

def main():
    def t_function_1(t, t0, beta=0.60):
        return t0*beta**t
    def t_function_2(t, t0, beta=0.3):
        return (t0-beta*t)
    def t_function_3(t, a=0.5, b=0.65):
        return (a/(log(t+b)))
    params = [{'steps': 10000, 'temperature': 10000, 't_function': t_function_2},
            #{'steps': 5000, 'temperature': 5000, 't_function': t_function_2}
            ]

    data = {'x_train': load('images/kmnist-train-imgs.npz'),
        'x_test': load('images/kmnist-test-imgs.npz'),
        'y_train': [str(i) for i in load('images/kmnist-train-labels.npz')],
        'y_test': [str(i) for i in load('images/kmnist-test-labels.npz')]}

    x_train, x_test, y_train, y_test = train_test_split(data['x_train'][:800], 
    data['y_train'][:800], test_size=0.33, stratify=data['y_train'][:800], random_state=42)
    
    data_2 = {'x_train': x_train, #1250
        'x_test': x_test, #500
        'y_train': y_train,
        'y_test': y_test}

    sa = SimulatedAnnealing(data=data_2, **params[0])
    sa_results = sa.simulate()
    with open('sa_t2_03', 'w') as f:
        json.dump(sa_results, f)

    return sa_results

if __name__ == '__main__':
    main()

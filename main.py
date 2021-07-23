from simulated_annealing import SimulatedAnnealing
import numpy as np
from sklearn.model_selection import train_test_split

def load(f):
        return np.load(f)['arr_0']

def main():
    def t_function_1(t, t0, beta=0.65):
        return t0*beta**t
    def t_function_2(t, t0, beta=0.65):
        return (t0-beta*t)
    params = [{'steps': 1000, 'temperature': 1000, 't_function': t_function_1},
            #{'steps': 10, 'temperature': 10, 't_function': t_function_2}
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

    annealing_results = list()
    for kwargs in params:
        print(kwargs)
        sa = SimulatedAnnealing(data=data_2, **kwargs)
        annealing_results.append(sa.simulate())

    return annealing_results

if __name__ == '__main__':
    print(main())

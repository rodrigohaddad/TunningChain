import wisardpkg as wp
import numpy as np
from sklearn.metrics import accuracy_score
from thermometer import ThermometerEncoder

clus_kwargs = {'resolution': list(range(3, 11)),
                'addressSize': list(range(3, 36)),
                'minScore': [x*0.1 for x in range(1, 9)],
                'threshold': list(range(3, 11)),
                'discriminatorLimit': list(range(3, 11))}

class WeighlessNetwok():  
    def __init__(self, data):
        x_test = data['x_test']
        x_train = data['x_train']
        self.y_train = data['y_train']
        self.y_test = data['y_test']
        self.x_train = list()
        self.x_test = list()
        for resolution in range(3, 11):
            self.x_train.append(self.simple_thermometer(x_train, resolution))
            self.x_test.append(self.simple_thermometer(x_test, resolution))

    def simple_thermometer(self, arr, resolution, minimum = 0, maximum = 255) -> list:
        therm = ThermometerEncoder(
            maximum=maximum, minimum=minimum, resolution=resolution)
        return [np.uint8(therm.encode(x)).flatten() for x in arr]

    def train(self, parameters):
        #resolution, addressSize, minScore, threshold, discriminatorLimit = parameters
        resolution, addressSize, minScore, threshold = parameters
        #clus = wp.ClusWisard(addressSize, minScore*0.1, threshold, discriminatorLimit)
        clus = wp.ClusWisard(addressSize, minScore*0.1, threshold, 6)
        clus.train(self.x_train[resolution-3], self.y_train)
        return clus.classify(self.x_test[resolution-3])

    def eval(self, y_pred):
        return accuracy_score(self.y_test, y_pred, normalize=False)
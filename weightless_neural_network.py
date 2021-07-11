import wisardpkg as wp
import numpy as np
from thermometer import ThermometerEncoder

clus_kwargs = {'resolution': list(range(3, 11)),
                'addressSize': list(range(3, 36)),
                'minScore': [x*0.1 for x in range(1, 9)],
                'threshold': list(range(3, 11)),
                'discriminatorLimit': list(range(3, 11))}

class WeighlessNetwok():  
    def __init__(self, data):
        #apply all the thermoter combinations
        pass

    def simple_thermometer(self, arr, minimum, maximum, resolution) -> list:
        therm = ThermometerEncoder(
            maximum=maximum, minimum=minimum, resolution=resolution)
        return [np.uint8(therm.encode(x)).flatten() for x in arr]

    def train(self):
        pass

    def eval(self):
        pass
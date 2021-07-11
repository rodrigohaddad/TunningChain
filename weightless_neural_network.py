import wisardpkg as wp
import numpy as np
from thermometer import ThermometerEncoder

class WeighlessNetwok():  
    def simple_thermometer(self, arr, minimum, maximum, resolution) -> list:
        therm = ThermometerEncoder(
            maximum=maximum, minimum=minimum, resolution=resolution)
        return [np.uint8(therm.encode(x)).flatten() for x in arr]
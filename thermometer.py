import numpy as np

class ThermometerEncoder(object):
    def __init__(self, minimum, maximum, resolution):
        self.minimum = minimum
        self.maximum = maximum
        self.resolution = resolution

    def encode(self, X):
        X = np.asarray(X)

        if X.ndim == 0:
            def f(i): return X > self.minimum + i * \
                (self.maximum - self.minimum)/self.resolution
        elif X.ndim == 1:
            def f(i, j): return X[j] > self.minimum + i * \
                (self.maximum - self.minimum)/self.resolution
        else:
            def f(i, j, k): return X[k, j] > self.minimum + \
                i*(self.maximum - self.minimum)/self.resolution
        return np.fromfunction(
            f,
            (self.resolution, *reversed(X.shape)),
            dtype=int
        ).astype(int)
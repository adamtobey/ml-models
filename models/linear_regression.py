import numpy as np

class LinearRegression(object):
    def __init__(self):
        self.weights = None

    def fit(self, inputs, targets):
        self.weights = np.linalg.lstsq(inputs, targets)[0]

    def predict(self, inputs):
        assert self.weights is not None, "Model must be trained before predicting"
        return inputs.dot(self.weights)

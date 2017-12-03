import numpy as np
from design_matrices import DesignMatrixTransformer, Regularizers, ScalarBasisFunctions

class LinearRegression(object):

    def __init__(self, basis_function=ScalarBasisFunctions.Identity(), regularizer=Regularizers.NoReg()):
        self.weights = None
        self.transformer = DesignMatrixTransformer(basis_function=basis_function, regularizer=regularizer)

    def fit(self, data_provider):
        inputs, targets = self.transformer.train(data_provider).next()
        self.weights = np.linalg.lstsq(inputs, targets)[0]

    def predict(self, inputs):
        assert self.weights is not None, "Model must be trained before predicting"
        return self.transformer.make_design_matrix(inputs).dot(self.weights)

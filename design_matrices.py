import numpy as np

class ScalarBasisFunctions(object):

    @staticmethod
    def Identity():
        return lambda x: x.reshape(1, -1), lambda _: 1

    @staticmethod
    def Polynomial(degree):
        return lambda x: np.concatenate([(x**d).reshape(1, -1) for d in range(degree + 1)]).T, lambda _: degree + 1

class BasisFunctions(object):

    @staticmethod
    def Identity():
        return lambda x: x, lambda x: x.shape[1:]

    @staticmethod
    def Affine():
        def affine(x):
            xd = x.shape[0]
            return np.concatenate([np.ones((xd, 1)), x], axis=1)
        return affine, lambda x: (x.shape[1] + 1,)

class Regularizers(object):

    @staticmethod
    def NoReg():
        return lambda *x: x

    @staticmethod
    def L2(l2_cost):
        rl2 = l2_cost ** 0.5
        def transform(inputs, targets):
            n_weights = inputs.shape[1]
            t_in = np.concatenate([inputs, np.diag([rl2 for _ in range(n_weights)])])
            t_out = np.concatenate([targets, [0 for _ in range(n_weights)]])
            return t_in, t_out
        return transform

class DesignMatrixTransformer(object):

    def __init__(self, basis_function, regularizer):
        self.basis_function, self.t_shape = basis_function
        self.regularize = regularizer

    def shape(self, data_provider):
        dummy, _ = data_provider.train().next()
        return self.t_shape(dummy)

    def train(self, data_provider):
        def iterate():
            for batch, targets in data_provider.train():
                yield self.transform(batch, targets)
        return iterate

    def make_design_matrix(self, inputs):
        return self.basis_function(inputs)

    def transform(self, inputs, targets):
        return self.regularize(self.make_design_matrix(inputs), targets)

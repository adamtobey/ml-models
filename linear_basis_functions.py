import numpy as np

class ScalarBasisFunctions(object):

    @staticmethod
    def Identity():
        return lambda x: x.reshape(1, -1)

    @staticmethod
    def Polynomial(degree):
        return lambda x: np.concatenate([(x**d).reshape(1, -1) for d in range(degree + 1)]).T

class BasisFunctions(object):

    @staticmethod
    def Identity():
        return lambda x: x

    @staticmethod
    def Affine():
        def affine(x):
            xd = x.shape[0]
            if xd == 0:
                return np.zeros((0, 0))
            else:
                return np.concatenate([np.ones((xd, 1)), x], axis=1)
        return affine

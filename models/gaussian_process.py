import numpy as np

class Kernels(object):

    @staticmethod
    def GaussianKernel(s2, l):
        lr = 1 / l.reshape((1, 1, -1))
        def eval_mat(X, Y):
            Lr = np.broadcast_to(lr, X.shape)
            return s2 * np.exp(-0.5 * (X - Y)**2 * Lr)
        return eval_mat

class GaussianProcessRegression(object):

    def __init__(self, kernel, s2y=0.1):
        self.kernel = kernel
        self.i_cov = np.identity(1)
        self.obs = None
        self.s2y = s2y

    def K(self, X, Y):
        assert X.shape[1] == Y.shape[1], "Input dimensionality must match"
        comp_shape = (X.shape[0], Y.shape[0], X.shape[1])
        x = np.broadcast_to(X.reshape((X.shape[0], 1, X.shape[1])), comp_shape)
        y = np.broadcast_to(Y.reshape((1, *Y.shape)), comp_shape)
        return self.kernel(x, y)[:,:,0]

    def fit(self, X, y):
        kk = self.K(X, X)
        si = self.s2y * np.identity(X.shape[0])
        self.i_cov = np.linalg.inv(kk + si)
        self.obs = X, y

    def predict(self, X):
        assert self.obs is not None, "Model must be fit before predicting"
        oX, oy = self.obs
        return self.K(X, oX).dot(self.i_cov).dot(oy)

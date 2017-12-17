import numpy as np

class BayesianLinearRegression(object):

    def __init__(self, prior_mean, prior_covariance, s2y=0.1):
        self.posterior_mean = prior_mean
        self.posterior_covariance = prior_covariance
        self.posterior_precision = np.linalg.inv(prior_covariance)
        self.s2y = s2y

    def fit(self, X, y):
        prior_precision = self.posterior_precision
        prior_mean = self.posterior_mean
        self.posterior_precision = (self.s2y * prior_precision + X.T.dot(X)) / self.s2y
        self.posterior_covariance = np.linalg.inv(self.posterior_precision)
        self.posterior_mean = self.posterior_covariance.dot(prior_precision).dot(prior_mean) + (self.posterior_covariance.dot(X.T).dot(y)) / self.s2y

    def predictive_params(self, X):
        means = X.dot(self.posterior_mean)
        vars = X.dot(self.posterior_covariance).dot(X.T).diagonal() + self.s2y
        return means, vars

    def predict_interval(self, X):
        means, vars = self.predictive_params(X)
        stds = vars ** 0.5
        return means - stds, means, means + stds

    def predict(self, X):
        return X.dot(self.posterior_mean)

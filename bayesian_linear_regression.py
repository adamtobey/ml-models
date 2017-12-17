from models.bayesian_linear_regression import BayesianLinearRegression
import numpy as np

regressor = BayesianLinearRegression(prior_mean=np.zeros((2,)), prior_covariance=0.4*np.identity(2))

x_train = np.array([
    [1, 0],
    [1, 1],
    [1, 2],
    [1, 3]
])
y_train = np.array([3.1, 5.4, 6.8, 9])

x_test = np.array([
    [1, 4],
    [1, 5],
])
y_test = np.array([11, 13])

regressor.fit(x_train, y_train)

print("Model weights: ", regressor.posterior_mean, regressor.posterior_covariance)
print("Model fit: ", regressor.predict(x_test))
print("Model interval: ", regressor.predict_interval(x_test))
print("Y test: ", y_test)

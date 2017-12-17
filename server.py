from bokeh.plotting import curdoc
from plots.bayesian_linear_regression import InteractiveBayesianLinearRegression
from models.bayesian_linear_regression import BayesianLinearRegression
import numpy as np

regressor = BayesianLinearRegression(prior_mean=np.zeros((2,)), prior_covariance=0.4*np.identity(2))
data = dict(x=[1], y=[1])
plot = InteractiveBayesianLinearRegression(data, regressor)

plot.render(curdoc())

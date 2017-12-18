from bokeh.plotting import curdoc
from plots.bayesian_polynomial_regression import InteractiveBayesianPolynomialRegression
import numpy as np

data = dict(x=[1], y=[1])
plot = InteractiveBayesianPolynomialRegression(data)

plot.render(curdoc())

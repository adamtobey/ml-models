from bokeh.plotting import curdoc
from plots.polynomial_regression import InteractivePolynomialRegression
import numpy as np

data = dict(x=[1], y=[1])
plot = InteractivePolynomialRegression(data)

plot.render(curdoc())

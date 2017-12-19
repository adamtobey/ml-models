from bokeh.plotting import curdoc
from plots.logistic_regression import InteractiveLogisticRegression
import numpy as np

data = [dict(x=[1], y=[1]), dict(x=[3], y=[3])]
plot = InteractiveLogisticRegression(data)

plot.render(curdoc())

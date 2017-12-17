from bokeh.plotting import curdoc
from plots.gaussian_process import InteractiveGaussianProcess
from models.gaussian_process import GaussianProcessRegression, Kernels
import numpy as np

data = dict(x=[1], y=[1])
plot = InteractiveGaussianProcess(data)

plot.render(curdoc())

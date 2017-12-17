from bokeh.plotting import curdoc
from plots.gaussian_process import InteractiveGaussianProcess
from models.gaussian_process import GaussianProcessRegression, Kernels
import numpy as np

regressor = GaussianProcessRegression(kernel=Kernels.GaussianKernel(1.2, np.array([0.5])))
data = dict(x=[1], y=[1])
plot = InteractiveGaussianProcess(data, regressor)

plot.render(curdoc())

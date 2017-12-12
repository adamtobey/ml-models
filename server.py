from bokeh.plotting import curdoc
from plots.linear_regression import InteractiveLinearRegression
from models.linear_regression import LinearRegression

regressor = LinearRegression()
data = dict(x=[1], y=[1])
plot = InteractiveLinearRegression(data, regressor)

plot.render(curdoc())

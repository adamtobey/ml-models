from bokeh.plotting import curdoc
from plots.linear_regression import InteractiveLinearRegression
from models.linear_regression import LinearRegression

regressor = LinearRegression()
plot = InteractiveLinearRegression(regressor)

plot.render(curdoc())

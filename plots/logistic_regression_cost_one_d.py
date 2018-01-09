import numpy as np
from bokeh.plotting import figure
from bokeh.layouts import row
from bokeh.models.callbacks import CustomJS
from models.logistic_regression import LogisticRegressionCost
from bokeh.models.widgets import RadioButtonGroup, Slider

class LogisticRegressionCost1D(object):

    PLOT_POINTS = 150
    X_RANGE = (-30, 30)
    Y_RANGE = (-5, 50)

    def __init__(self):
        self.fig = figure(x_range=self.X_RANGE, y_range=self.Y_RANGE, tools="", toolbar_location=None)

        self.cost = LogisticRegressionCost()

        self.pos_cost = self.fig.line(x=[], y=[], color="#FF8F00")
        self.neg_cost = self.fig.line(x=[], y=[], color="#008FFF")

        self.scatters = dict(
            red = self.fig.scatter(x=[], y=[], color="#FF0000"),
            blue = self.fig.scatter(x=[], y=[], color="#0000FF")
        )

        radiokeys = list(self.scatters.keys())
        self.which_scatter = RadioButtonGroup(labels=radiokeys, active=0)
        data_sources = {
            key: scatter.data_source
            for key, scatter in self.scatters.items()
        }
        js_argdict = f"""{{
            { ','.join(f"{i}: {k}" for i, k in enumerate(radiokeys)) }
        }}"""
        self.fig.js_on_event('tap', CustomJS(
            args=dict(which_scatter=self.which_scatter, **data_sources),
            code=f"""
                var argdict = {js_argdict};
                var scatter = argdict[which_scatter.active];
                var data = {{
                    'x': scatter.data.x,
                    'y': scatter.data.y
                }};
                data['x'].push(cb_obj['x']);
                data['y'].push(0);
                scatter.data = data;
                scatter.change.emit();
            """
        ))

        self.renderable = row(self.fig, self.which_scatter)

        for data_source in data_sources.values():
            data_source.on_change('data', self.update)

    def plot_cost(self, plot, x_weight, X, t):
        ws = np.linspace(*self.X_RANGE, self.PLOT_POINTS)
        W = np.ndarray((self.PLOT_POINTS, 2))
        W[:,0] = ws
        W[:,1] = x_weight
        y_min, y_max = self.Y_RANGE
        cut = lambda x: max(y_min - 1, min(y_max + 1, x))
        cost = [cut(self.cost.eval(X, t, w)) for w in W]
        plot.data_source.data = dict(x=-ws/x_weight, y=cost)

    def update(self, attr, old, new):
        t = []
        X = []
        for i, scatter in enumerate(self.scatters.values()):
            x = scatter.data_source.data['x']
            X.extend([np.array([1, p]) for p in x])
            t.extend([i for _ in range(len(x))])
        t = np.array(t)
        X = np.array(X)

        # plot cost
        self.plot_cost(self.pos_cost, 1, X, t)
        self.plot_cost(self.neg_cost, -1, X, t)

    def render(self, doc):
        doc.add_root(self.renderable)

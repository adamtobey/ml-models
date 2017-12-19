from bokeh.plotting import figure
from bokeh.models.callbacks import CustomJS
from bokeh.layouts import row, column
from bokeh.models.widgets import RadioGroup

class InteractivePlot(object):

    def __init__(self, data=dict(x=[], y=[])):
        self.figure = figure(x_range=(0,10), y_range=(0,10), toolbar_location=None)
        self._initialize_figure(data)

    def _initialize_figure(self, data):
        self.scatter = self.figure.scatter(**data)
        data_source = self.scatter.data_source
        self.figure.js_on_event('tap', CustomJS(args=dict(source=data_source), code="""
            var data = {
                'x': source.data.x,
                'y': source.data.y
            };
            data['x'].push(cb_obj['x']);
            data['y'].push(cb_obj['y']);
            source.data = data;
            source.change.emit();
        """))
        self.initialize_figure(self.figure, self.scatter)
        data_source.on_change('data', self._update_figure)

    def initialize_figure(self, figure, scatter):
        pass

    def update_figure(self, figure, scatter, new_data, old_data):
        pass

    def _update_figure(self, attr, old, new):
        self.update_figure(self.figure, self.scatter, new, old)

    def render(self, doc):
        doc.add_root(self.figure)

class InteractiveParametricPlot(InteractivePlot):

    def __init__(self, data=dict(x=[], y=[])):
        self.params = self.parameters()
        for name, param in self.params.items():
            param.on_change('value', self.update_param)
        super().__init__(data)

    def update_param(self, attr, old, new):
        self.update_figure(self.figure, self.scatter, new, old)

    def parameters(self):
        return {}

    def render(self, doc):
        doc.add_root(row(self.figure, column(*self.params.values())))

class MulticlassParametricPlot(InteractiveParametricPlot):

    def _initialize_figure(self, data):
        self.scatters = self.make_scatters(data)
        radiokeys = list(self.scatters.keys())
        self.which_scatter = RadioGroup(labels=radiokeys, active=0)
        data_sources = {
            key: scatter.data_source
            for key, scatter in self.scatters.items()
        }
        self.figure.js_on_event('tap', CustomJS(args=dict(which_scatter=self.which_scatter, **data_sources), code="""
            var argdict = {};
            var scatter = argdict[which_scatter.active];
            var data = {{
                'x': scatter.data.x,
                'y': scatter.data.y
            }};
            data['x'].push(cb_obj['x']);
            data['y'].push(cb_obj['y']);
            scatter.data = data;
            scatter.change.emit();
        """.format('{{{}}}'.format(','.join(["{}: {}".format(i, k) for i, k in enumerate(radiokeys)])))))
        self.initialize_figure(self.figure, self.scatters)
        for scatter in self.scatters.values():
            scatter.data_source.on_change('data', self._update_figure)

    def _update_figure(self, attr, old, new):
        self.update_figure(self.figure, self.scatters, new, old)

    def make_scatters(self, data):
        return {}

    def render(self, doc):
        doc.add_root(row(self.figure, column(self.which_scatter, *self.params.values())))

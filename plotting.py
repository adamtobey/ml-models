from bokeh.plotting import figure
from bokeh.models.callbacks import CustomJS

class InteractivePlot(object):

    def __init__(self, data=dict(x=[], y=[])):
        self._initialize_figure(data)

    def _initialize_figure(self, data):
        self.figure = figure(x_range=(0,10), y_range=(0,10), toolbar_location=None)
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

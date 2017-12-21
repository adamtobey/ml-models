from bokeh.plotting import figure
from bokeh.models.callbacks import CustomJS
from bokeh.layouts import column, row
from bokeh.models.widgets import RadioButtonGroup, Slider


X_RANGE = Y_RANGE = (-5, 10)


class BaseInteractivePlot(object):

    BASE_FIGURE_PARAMS = dict(
        x_range = X_RANGE,
        y_range = Y_RANGE,
        tools = "",
        toolbar_location = None
    )

    def __init__(self, **figure_params):
        self.figure = figure(**{**self.BASE_FIGURE_PARAMS, **figure_params})

        self._change_listeners = []
        self._data_template = {}

        self._setup_plot()

    def _setup_plot(self):
        pass

    def _handle_state_change(self, attr, new, old):
        state = self.get_state()
        for listener in self._change_listeners:
            listener(state)

    def register_state_attribute(self, name, model, attribute, prefix=[]):
        model.on_change(attribute, self._handle_state_change)

        template = self._data_template
        for pre in prefix:
            if pre not in template:
                template[pre] = {}
            template = template[pre]

        template[name] = lambda: model.__getattribute__(attribute)

    def get_state(self):
        def evaluate_state(node):
            if type(node) is not dict:
                return node()
            else:
                return {
                    name: evaluate_state(child) for name, child in node.items()
                }
        return evaluate_state(self._data_template)

    def add_change_listener(self, listener):
        self._change_listeners.append(listener)

    def drawable(self):
        return self.figure


class SingleClassPlot(BaseInteractivePlot):

    def _setup_plot(self):
        self._scatter = self.figure.scatter(x=[], y=[])

    def enable_interaction(self):
        data_source = self._scatter.data_source
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
        self.register_state_attribute('inputs', data_source, 'data')


class MultiClassPlot(BaseInteractivePlot):

    def _setup_plot(self):
        self._scatters = {}

    def add_scatter(self, name, **scatter_opts):
        self._scatters[name] = self.figure.scatter(**scatter_opts)

    def enable_interaction(self):
        radiokeys = list(self._scatters.keys())
        self._which_scatter = RadioButtonGroup(labels=radiokeys, active=0)
        data_sources = {
            key: scatter.data_source
            for key, scatter in self._scatters.items()
        }
        js_argdict = f"""{{
            { ','.join(f"{i}: {k}" for i, k in enumerate(radiokeys)) }
        }}"""
        self.figure.js_on_event('tap', CustomJS(
            args=dict(which_scatter=self._which_scatter, **data_sources),
            code=f"""
                var argdict = {js_argdict};
                var scatter = argdict[which_scatter.active];
                var data = {{
                    'x': scatter.data.x,
                    'y': scatter.data.y
                }};
                data['x'].push(cb_obj['x']);
                data['y'].push(cb_obj['y']);
                scatter.data = data;
                scatter.change.emit();
            """
        ))

        for source_name, data_source in data_sources.items():
            self.register_state_attribute(source_name, data_source, 'data', prefix=['inputs'])

    def drawable(self):
        return column(self._which_scatter, self.figure)


class ParametricPlotContainer(object):

    def __init__(self, plot):
        self._plot = plot
        self._params = []

    def slider(self, title, **slider_params):
        slider = Slider(title=title, **slider_params)
        self._params.append(slider)
        self._plot.register_state_attribute(title, slider, 'value')

    def drawable(self):
        return row(self._plot.drawable(), column(*self._params))

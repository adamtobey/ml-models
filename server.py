from bokeh.plotting import figure, curdoc
from bokeh.models.callbacks import CustomJS
import numpy as np
from design_matrices import ScalarBasisFunctions

p = figure(x_range=(0,10), y_range=(0,10), toolbar_location=None)
print(p.js_on_event)

r = p.scatter(x=[0], y=[1])
m, b = 1, 0
xs = np.array([-1000,1000])
ys = xs * m + b
l = p.line(x=xs, y=ys)

ds = r.data_source
lds = l.data_source

p.js_on_event('tap', CustomJS(args=dict(source=ds), code="""
    console.log(cb_obj);
    var data = {
        'x': source.data.x,
        'y': source.data.y
    };
    data['x'].push(cb_obj['x']);
    data['y'].push(cb_obj['y']);
    source.data = data
    source.change.emit();
"""))

def test(attr, old, new):
    global m, b, lds, xs
    b, m = regression(new)
    lds.data = {
        'x': xs,
        'y': xs * m + b
    }

basis = ScalarBasisFunctions.Polynomial(1)[0]

def regression(new):
    x, y = basis(np.array(new['x'])), np.array(new['y'])
    w = np.linalg.lstsq(x, y)[0]
    return w

ds.on_change('data', test)

curdoc().add_root(p)

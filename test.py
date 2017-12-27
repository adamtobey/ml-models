import numpy as np
<<<<<<< HEAD
=======
from calculus import gradient
>>>>>>> 4e75448... wip
from models.logistic_regression import LogisticRegressionCost, RefLRC

x = np.arange(4).reshape((2,2))
w = y = np.arange(2)

r = RefLRC()
t = LogisticRegressionCost()

<<<<<<< HEAD
=======
anal =

>>>>>>> 4e75448... wip
print("Ref gradient: ", r.gradient(x, y, w))
print("Test gradient: ", t.gradient(x, y, w))
print("Ref hessian: ", r.hessian(x, y, w))
print("Test hessian: ", t.hessian(x, y, w))

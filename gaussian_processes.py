from models.gaussian_process import GaussianProcessRegression, Kernels
import numpy as np
import matplotlib.pyplot as plt

regressor = GaussianProcessRegression(kernel=Kernels.GaussianKernel(1.2, np.array([0.5])))

inputs = np.arange(0, 3.14, 0.2).reshape((-1, 1))
outputs = np.sin(inputs)

x_test = np.arange(0, 3.14, 0.2).reshape((-1, 1)) + 0.1
y_test = np.sin(x_test)

regressor.fit(inputs, outputs)

predictions = regressor.predict(x_test)

fig, ax = plt.subplots()
ax.plot(x_test, y_test, label="Test set")
ax.plot(x_test, predictions, label="Gaussian Process")
ax.legend()
fig.savefig("GaussianProcess.png")

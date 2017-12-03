from models.logistic_regression import LogisticRegression, BasicDataProvider
import numpy as np

inputs = np.array([
    [-1, -1],
    [-1, -0.2],
    [-1, -0.7],
    [3, 4],
    [2, 5],
    [3, 3]
], dtype='float64')

targets = np.array([
    0,
    0,
    0,
    1,
    1,
    1
])

x_test = np.array([
    [-0.1, -0.2],
    [2.3, 4.9]
])

y_test = np.array([0, 1])

data_provider = BasicDataProvider(inputs, targets)

classifier = LogisticRegression()

classifier.fit(data_provider, 50)


print("Predictions: ", classifier.predict(x_test))
print("Y Test: ", y_test)

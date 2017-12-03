from models.logistic_regression import LogisticRegression
from data_providers import BasicDataProvider
import numpy as np

inputs = np.array([
    [0, 0],
    [0, 1],
    [0, 2],
    [0, 10],
    [0, 11],
    [0, 12]
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
    [0, 1.2],
    [0, 11.2]
])

y_test = np.array([0, 1])

data_provider = BasicDataProvider(inputs, targets)

classifier = LogisticRegression()

classifier.fit(data_provider, 50)


print("Predictions: ", classifier.predict(x_test))
print("Y Test: ", y_test)

print("Weights: ", classifier.weights)

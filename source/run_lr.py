#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LogisticRegression1 test runner

Chapter "Logistic Regression"
"""

# imports
import numpy as np
from lr import LogisticRegression

# load data
data = np.genfromtxt('iris2.tsv',
                     dtype=[('X', float, 4), ('y', int)])

# learn model
clr = LogisticRegression()
clr.fit(data['X'], data['y'])

# test model
predict_y = clr.predict(data['X'])

# print results
for i in range(0, 100, 10):
    print(i, data['y'][i], predict_y[i])

# print accuracy
print("Accuracy =", end=' ')
print(np.sum(data['y'] == predict_y, dtype=float) / predict_y.shape[0])

# print parameters
print("coef =", clr.coef_)
print("intercept_ =", clr.intercept_)

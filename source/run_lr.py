#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LogisticRegression1 test runner

Chapter "Logistic Regression"
"""

# imports
import numpy as np
from sklearn.linear_model import LogisticRegression

# load data
data = np.genfromtxt('iris2.tsv',
                     dtype=[('X', 'f', 4), ('y', 'i')])

# learn model
clr = LogisticRegression()
clr.fit(data['X'], data['y'])

# test model
predict_y = clr.predict(data['X'])

# print results
for i in xrange(0, 100, 10):
    print i, data['y'][i], predict_y[i]

# print accuracy
print "Accuracy =",
print np.sum(data['y'] == predict_y, dtype=float) / predict_y.shape[0]
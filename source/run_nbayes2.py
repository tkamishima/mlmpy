#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NaiveBayes2 test runner

Chapter "Naive Bayes: an Advanced Course"
"""

# imports
import numpy as np
from nbayes2 import NaiveBayes1, NaiveBayes2

# load data
data = np.genfromtxt('vote_filled.tsv', dtype=int)

# split data
X = data[:, :-1]
y = data[:, -1]

# Naive1Bayes1

# learn model
clr = NaiveBayes1()
clr.fit(X, y)

# test model
predict_y = clr.predict(X[:10, :])

# print results
print("NaiveBayes1")
for i in range(10):
    print((i, y[i], predict_y[i]))

# Naive1Bayes2

# learn model
clr = NaiveBayes2()
clr.fit(X, y)

# test model
predict_y = clr.predict(X[:10, :])

# print results
print("NaiveBayes2")
for i in range(10):
    print((i, y[i], predict_y[i]))


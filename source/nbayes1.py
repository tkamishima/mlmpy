#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NaiveBayes1 class

Chapter "Naive Bayes: a Primary Course"
"""

# imports
import numpy as np

# public symbols
__all__ = ['NaiveBayes1']

class NaiveBayes1(object):
    """
    Naive Bayes class (1)

    Attributes
    ----------
    `pY_` : array_like, shape=(n_classes), dtype=float
        pmf of a class
    `pXgY_` : array_like, shape(n_features, n_classes, n_fvalues), dtype=float
        pmf of feature values given a class
    """

    def __init__(self):
        self.pY_ = None
        self.pXgY_ = None

    def fit(self, X, y):
        """
        Fitting model

        Parameters
        ----------
        X : array_like, shape=(n_samples, n_features), dtype=int
            feature values of training samples
        y : array_like, shape=(n_samples), dtype=int
            class labels of training samples
        """

        # constants
        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_classes = 2
        n_fvalues = 2

        # check the size of y
        if n_samples != len(y):
            raise ValueError('Mismatched number of samples.')

        # count up n[yi=y]
        nY = np.zeros(n_classes, dtype=int)
        for i in range(n_samples):
            nY[y[i]] += 1

        # calc pY_
        self.pY_ = np.empty(n_classes, dtype=float)
        for i in range(n_classes):
            self.pY_[i] = nY[i] / n_samples

        # count up n[x_ij=xj, yi=y]
        nXY = np.zeros((n_features, n_fvalues, n_classes), dtype=int)
        for i in range(n_samples):
            for j in range(n_features):
                nXY[j, X[i, j], y[i]] += 1

        # calc pXgY_
        self.pXgY_ = np.empty((n_features, n_fvalues, n_classes),
                              dtype=float)
        for j in range(n_features):
            for xi in range(n_fvalues):
                for yi in range(n_classes):
                    self.pXgY_[j, xi, yi] = nXY[j, xi, yi] / nY[yi]

    def predict(self, X):
        """
        Predict class

        Parameters
        ----------
        X : array_like, shape=(n_samples, n_features), dtype=int
            feature values of unseen samples

        Returns
        -------
        y : array_like, shape=(n_samples), dtype=int
            predicted class labels
        """

        # constants
        n_samples = X.shape[0]
        n_features = X.shape[1]

        # memory for return values
        y = np.empty(n_samples, dtype=int)

        # for each feature in X
        for i, xi in enumerate(X):

            # calc joint probability
            logpXY = np.log(self.pY_) + \
                     np.sum(np.log(self.pXgY_[np.arange(n_features), xi, :]),
                            axis=0)

            # predict class
            y[i] = np.argmax(logpXY)

        return y

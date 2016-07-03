#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LogisticRegression class

Chapter "Logistic Regression"

Checking a range of inputs to a :meth:`sigmoid` function.
"""

# imports
import numpy as np

# public symbols
__all__ = ['LogisticRegression']


class LogisticRegression(object):
    """
    Logistic Regression class

    Parameters
    ----------
    C : float
        regularization parameter

    Attributes
    ----------
    `coef_` : array_like, shape=(n_features + 1), dtype=float
        Weight coefficients of linear model
    `intercept_` : float
        intercept parameter
    `n_features_` : int
        number of features
    `n_samples_` : int
        number of training data
    """

    @staticmethod
    def sigmoid(x):
        """
        sigmoid function

        implementation with input range check

        Parameters
        ----------
        x : array_like, shape=(n_data), dtype=float
            arguments of function

        Returns
        -------
        sig : array, shape=(n_data), dtype=float
            1.0 / (1.0 + exp(- x))
        """
        sigmoid_range = 34.538776394910684

        if x <= -sigmoid_range:
            return 1e-15
        if x >= sigmoid_range:
            return 1.0 - 1e-15

        return 1.0 / (1.0 + np.exp(-x))

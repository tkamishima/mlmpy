#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LogisticRegression class

Chapter "Logistic Regression"
"""

# imports
import numpy as np
from scipy.optimize import minimize

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

    def __init__(self, C=1.0):
        self.C = C

        self.coef_ = None
        self.intercept_ = 0
        self.n_features_ = 0
        self.n_samples_ = 0

        self._param_dtype = np.dtype([])

    @staticmethod
    def sigmoid(x):
        """
        sigmoid function

        Parameters
        ----------
        x : array_like, shape=(n_data), dtype=float
            arguments of function

        Returns
        -------
        sig : array, shape=(n_data), dtype=float
            1.0 / (1.0 + exp(- x))
        """

        # restrict domain of sigmoid function within [1e-15, 1 - 1e-15]
        sigmoid_range = 34.538776394910684
        x = np.clip(x, -sigmoid_range, sigmoid_range)

        return 1.0 / (1.0 + np.exp(-x))

    def loss(self, params, X, y):
        """

        Parameters
        ----------
        params : array, shape=(n_features + 1), dtype=float
            arguments of loss function
        X : array, shape=(n_samples, n_features), dtype=float
            feature values of training samples
        y : array, shape=(n_samples), dtype=int
            class labels of training samples

        Returns
        -------
        loss : float
            return value loss function
        """

        # decompose parameters
        coef = params.view(self._param_dtype)['coef'][0, :]
        intercept = params.view(self._param_dtype)['intercept'][0]

        # predicted probabilities of data
        p = self.sigmoid(np.dot(X, coef) + intercept)

        # likelihood
        # \sum_{x,s,y in D} (1 - y) log(1 - sigma) + y log(sigma)
        l = np.sum((1.0 - y) * np.log(1.0 - p) + y * np.log(p))

        # L2 regularizer
        r = np.sum(coef * coef) + intercept * intercept

        return - l + 0.5 * self.C * r

    def grad_loss(self, params, X, y):
        """

        Parameters
        ----------
        params : array, shape=(n_params,), dtype=float
            arguments of loss function
        X : array, shape=(n_samples, n_features), dtype=float
            feature values of training samples
        y : array, shape=(n_samples), dtype=int
            class labels of training samples

        Returns
        -------
        grad : array, shape=(n_params,), dtype=float
            return a gradient vector of a loss function
        """

        # decompose parameters
        coef = params.view(self._param_dtype)['coef'][0, :]
        intercept = params.view(self._param_dtype)['intercept'][0]

        # create empty gradient
        grad = np.empty_like(params)
        grad_coef = grad.view(self._param_dtype)['coef']
        grad_intercept = grad.view(self._param_dtype)['intercept']

        # predicted probabilities of data
        p = self.sigmoid(np.dot(X, coef) + intercept)

        # gradient of weight coefficients
        grad_coef[0, :] = np.dot(p - y, X) + self.C * coef

        # gradient of an intercept
        grad_intercept[0] = np.sum(p - y) + self.C * intercept

        return grad

    def fit(self, X, y):
        """
        Fitting model

        Parameters
        ----------
        X : array_like, shape=(n_samples, n_features), dtype=float
            feature values of training samples
        y : array_like, shape=(n_samples), dtype=int
            class labels of training samples
        """

        # constants
        self.n_samples_ = X.shape[0]
        self.n_features_ = X.shape[1]

        # dtype for model parameters to optimize
        # the top n_features of elements in parameters are weight coefficients
        # the last element of parameters is an intercept
        self._param_dtype = np.dtype([
            ('coef', float, self.n_features_),
            ('intercept', float)
        ])

        # optimize
        res = minimize(fun=self.loss,
                       x0=np.zeros(self.n_features_ + 1, dtype=float),
                       jac=self.grad_loss,
                       args=(X, y),
                       method='CG')

        # get result
        self.coef_ = res.x.view(self._param_dtype)['coef'][0, :].copy()
        self.intercept_ = res.x.view(self._param_dtype)['intercept'][0]

    def predict(self, X):
        """
        Predict class

        Parameters
        ----------
        X : array_like, shape=(n_samples, n_features), dtype=float
            feature values of unseen samples

        Returns
        -------
        y : array_like, shape=(n_samples), dtype=int
            predicted class labels
        """

        # predicted probabilities of data
        p = self.sigmoid(np.sum(X * self.coef_[np.newaxis, :], axis=1) +
                         self.intercept_)

        return np.where(p < 0.5, 0, 1)

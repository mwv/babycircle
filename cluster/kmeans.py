#!/usr/bin/python
# -*- coding: utf-8 -*-
""" 
"""

from __future__ import division

import numpy as np
import scipy.spatial

__author__ = 'Maarten Versteegh'

class KMeans(object):
    """Simple implementation of the K-Means algorithm."""
    def __init__(self, init_centers=None, n_clusters=2,
                 max_iter=100, tol=1e-4, seed=42):
        """Simple implementation of the K-Means algorithm.

        Arguments:
        init_centers : ndarray (n_clusters, n_features), optional
          initial cluster centers
          if not specified, random means are chosen
        n_clusters : int, optional, default=2
          number of clusters
        max_iter : int, optional, default=100
          maximum number of iterations
        tol : float, optional, default=1e-4
          tolerance for convergence
        seed : int
          seed for the random number generator
        """
        self.init_centers = init_centers
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

    def _check_fit_data(self, X):
        if X.shape[0] < self.n_clusters:
            raise ValueError('n_samples={0} should be >= k={1}'.format(X.shape[0],
                                                                       self.n_clusters))
        return X

    def _check_means(self, ndims):
        """Check if self.init_centers is of correct dimensionality or set it if it is None
        """
        if self.init_centers is None:
            # no means passed in, set randomly
            self.init_centers = np.random.random((self.n_clusters, ndims))
        else:
            if self.init_centers.shape[1] != ndims:
                raise ValueError('init_centers are {0}-dimensional, should be {1}-dimensional'\
                .format(self.init_centers.shape[1], ndims))
        return self.init_centers

    def _check_fitted(self):
        if not hasattr(self, 'centers_'):
            raise AttributeError('Model has not been trained yet.')

    def _check_test_data(self, X):
        _, n_dims = X.shape
        n_dims_exp = self.centers_.shape[1]
        if not n_dims == n_dims_exp:
            raise ValueError('Incorrect number of features. '
                             'Data has {0} features, expected {1}.'.format(n_dims,
                                                                           n_dims_exp))
        return X

    def fit(self, X):
        """Fit the model to the data

        Arguments:
        X : ndarray (n_samples, n_features)
          data

        Returns:
        self : KMeans object
          fitted model
        """
        nsamples, ndims = X.shape
        self.random_state = np.random.RandomState(self.seed)
        self.init_centers = self._check_means(ndims)

        X = self._check_fit_data(X)

        centers = self.init_centers

        for i in range(self.max_iter):
            centers_old = centers.copy()

            # assign labels: e-step
            labels, inertia = self._e_step(X, centers)

            # recompute means: m-step
            centers = self._m_step(X, labels)

            if np.sum((centers_old - centers) ** 2) < self.tol:
                break
        self.centers_ = centers
        self.labels_ = labels
        self.inertia_ = inertia
        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to

        Arguments:
        X : ndarray (n_samples, n_features)
          input data

        Returns:
        labels : ndarray (n_samples,)
          index of closest center each sample belongs to
        inertia : float
          distortion measure
        """
        self._check_fitted()
        X = self._check_test_data(X)
        return self._e_step(X, self.centers_)[0]

    def _e_step(self, X, centers, distances=None):
        """E-step, assign labels to each data point
        """
        n_samples = X.shape[0]
        if distances is None:
            distances = scipy.spatial.distance.cdist(X,centers)**2
        labels = np.argmin(distances, axis=1)
        inertia = np.sum(distances[np.arange(n_samples), labels])
        return labels, inertia

    def _m_step(self, X, labels):
        """M-step, recompute cluster centers
        """
        return np.array([X[labels==n].mean(axis=0) for n in np.unique(labels)])
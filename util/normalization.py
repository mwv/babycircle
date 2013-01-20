#!/usr/bin/python
# -*- coding: utf-8 -*-
""" 
"""

from __future__ import division

__author__ = 'Maarten Versteegh'

def normalize(X):
    """Mean-variance normalization

    Arguments:
    X : ndarray (nsamples, nfeatures)
      input array

    Returns:
    X' : ndarray (nsamples, nfeatures)
      normalized output array
    """
    return (X-X.mean(axis=0))/X.std(axis=0)
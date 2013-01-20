#!/usr/bin/python
# -*- coding: utf-8 -*-
""" 
"""

from __future__ import division

__author__ = 'Maarten Versteegh'

import os

import numpy as np

from cluster.kmeans import KMeans
from visualization.plotting import plotscatterhist
from util.normalization import normalize

def make_plots():
    """make scatter plots of each e-step and m-step of the kmeans algorithm
    uses old faithful dataset.
    """
    X = np.loadtxt(os.path.join('data', 'faithful.txt'))
    X = normalize(X)

    init_means = np.array([[-1.75,1],[1.75,-1]])

    clf = KMeans(init_centers=init_means, n_clusters=2)

    tmpdir = 'tmp'
    try:
        os.makedirs(tmpdir)
    except:
        pass
    basename = os.path.join(tmpdir,'faithful_kmeans')
    centers = init_means

    plotscatterhist(X, [], '{0}_{1}.png'.format(basename, 0),
                    centers, lumped=True, dpi=200)

    for i in range(1,50,2):
        print 'iteration:', i,
        centers_old = centers.copy()
        labels, inertia = clf._e_step(X, centers)
        print 'inertia:', inertia
        plotscatterhist(X, labels, '{0}_{1}.png'.format(basename, i),
                        centers, lumped=False, dpi=200)
        centers = clf._m_step(X, labels)
        plotscatterhist(X, labels, '{0}_{1}.png'.format(basename, i+1),
                        centers, lumped=False, dpi=200)
        if np.sum((centers_old - centers) ** 2) < 1e-2:
            break
    print 'starting video encoding...',
    print 'done.'

if __name__ == '__main__':
    make_plots()
#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Plot two-dimensional data as scatter plot with histogram on the sides.

 Usage:
 TODO insert usage example
 >>>

"""

from __future__ import division

__author__ = 'Maarten Versteegh'

import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from colors import generate_colors

def plotscatterhist(X, y,
                    outfile,
                    centers=None,
                    lumped=True,
                    dpi=600, transparent=False,
                    xlabel='', ylabel='', title=''):
    """Plot two-dimensional data as scatter plot with histograms on the sides.

    Arguments:
    X : ndarray (n_samples, 2)
      array holding the data with samples on the rows and features on the columns
    y : ndarray (n_samples,)
      array holding the class labels corresponding to X
    outfile : string
      filename to write image to
    lumped : bool
      if true: plot data without colors for the classes, otherwise with
    dpi : int
      dpi of output image
    transparent : bool
      make output image transparent or not
    xlabel : string
      label for the x-axis
    ylabel : string
      label for the y-axis
    title : string
      label for the image
    """
    nullfmt = NullFormatter()

    left,   width  = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx   = [left, bottom_h, width, 0.2]
    rect_histy   = [left_h, bottom, 0.2, height]

    fig = plt.figure(1, figsize=(8,8))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    axHistx.xaxis.set_major_formatter(nullfmt)
    axHistx.yaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    axHisty.xaxis.set_major_formatter(nullfmt)

    colors = ['r', 'b']

    if lumped:
        axScatter.scatter(X[:,0],X[:,1], color='grey', alpha=0.3)
    else:
#        colors = generate_colors(np.unique(y).shape[0])
        for y_idx, y_val in enumerate(np.unique(y)):
            axScatter.scatter(X[(y==y_val),0], X[(y==y_val),1], color=colors[y_idx], alpha=0.3)
    if not centers is None:
#        colors = generate_colors(centers.shape[0])
        for idx, c in enumerate(centers):
            axScatter.scatter(c[0], c[1], color=colors[idx], s=360, marker='x')

    binwidth = 0.1
    xymax = np.max([np.max(np.fabs(X[:,0])), np.max(np.fabs(X[:,1]))])
    lim = (int(xymax/binwidth) + 1) * binwidth

    axScatter.set_xlim((-lim, lim))
    axScatter.set_ylim((-lim, lim))


    bins = np.arange(-lim, lim+binwidth, binwidth)
    if lumped:
        axHistx.hist(X[:,0], bins=bins, normed=True, color='grey', alpha=0.5)
        axHisty.hist(X[:,1], bins=bins, orientation='horizontal', normed=True, color='grey', alpha=0.5)
    else:
#        colors = generate_colors(np.unique(y).shape[0])
        for y_idx, y_val in enumerate(np.unique(y)):
            axHistx.hist(X[(y==y_val),0],
                         color=colors[y_idx],
                         alpha=0.5)
            axHisty.hist(X[(y==y_val),1],
                         color=colors[y_idx],
                         alpha=0.5,
                         orientation='horizontal')

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    axScatter.set_xlabel(xlabel)
    axScatter.set_ylabel(ylabel)
    axScatter.set_title(title)

    plt.savefig(outfile, dpi=dpi, transparent=transparent)
    fig.clf()

#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Convenience functions for generating distinct colors.

Usage:
>>> generate_colors(4)
[(1.0, 0.0, 0.0), (0.5, 1.0, 0.0), (0.0, 1.0, 1.0), (0.5, 0.0, 1.0)]
"""

from __future__ import division

__author__ = 'Maarten Versteegh'

import math

def _hsv_to_rgb(h,f):
    """Convert a color specified by h-value and f-value to rgb triple
    """
    v = 1.0
    p = 0.0
    if h == 0:
        return v, f, p
    elif h == 1:
        return 1-f, v, p
    elif h == 2:
        return p, v, f
    elif h == 3:
        return p, 1-f, v
    elif h == 4:
        return f, p, v
    elif h == 5:
        return v, p, 1-f

def generate_colors(n):
    """Generate n distinct colors as rgb triples

    Arguments:
    n : int
      number of colors to generate

    Returns:
    List of rgb triples
    """
    hues = [360/n*i for i in range(n)]
    hs = [(math.floor(hue/60) % 6) for hue in hues]
    fs = [(hue/60 - math.floor(hue / 60)) for hue in hues]
    return [_hsv_to_rgb(h,f) for h,f in zip(hs,fs)]


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

# -> Manuscrit 415.41025
# -> Notes 418.25368


def get_figsize(columnwidth=415.41025, wf=0.5, hf=(5.**0.5 - 1.0) / 2.0):
    """Parameters:
      - wf [float]:  width fraction in columnwidth units
      - hf [float]:  height fraction in columnwidth units.
                     Set by default to golden ratio.
      - columnwidth [float]: width of the column in latex. Get this from LaTeX 
                             using \showthe\columnwidth
    Returns:  [fig_width,fig_height]: that should be given to matplotlib
    """
    fig_width_pt = columnwidth * wf
    inches_per_pt = 1.0 / 72.27               # Convert pt to inch
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = fig_width * hf      # height in inches
    return [fig_width, fig_height]


params = {# 'backend': 'pgf',
          'axes.labelsize': 10,
          'axes.titlesize': 11,
          'image.cmap': u'viridis'}  # extend as needed

plt.rc('text.latex', preamble=(r'\usepackage{amsmath} \usepackage{amssymb}'))
# mpl.use('pgf')
plt.style.use('seaborn')
plt.rc('font', **{'family': 'serif',
                  'serif': ['Computer Modern Roman']})
plt.rcParams.update(params)

plt.rc('text', usetex=True)
col_half = get_figsize()
col_full = get_figsize(wf=1.0)
col_3quarter = get_figsize(wf=.75)

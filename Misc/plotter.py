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


params = {'backend': 'pgf',
          'axes.labelsize': 10,
          'axes.titlesize': 11,
          'image.cmap': u'viridis'}  # extend as needed


mpl.use('pgf')
# pgf_with_rc_fonts = {
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"],                   # use latex default serif font
#     "font.sans-serif": ["DejaVu Sans"], # use a specific sans-serif font
# }
# mpl.rcParams.update(pgf_with_rc_fonts)
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.style.use('seaborn')
plt.rc('font', **{'family': 'serif',
                  'serif': ['Computer Modern Roman']})
plt.rcParams.update(params)
plt.rc('text', usetex=True)
col_half = get_figsize()
col_full = get_figsize(wf=1.0)


## PDF and CDF of rv X
plt.figure(figsize=col_half)
ax = plt.subplot(1, 1, 1)
ax.set_xlabel(r'$x$')
ax.set_xlim([-0.5, 6])
ax.set_ylim([-0.05, 1.05])

ax.plot([2, 4], [1 / 4.0, 1.0 / 4.0], 'k', label=r'$p_X$')
ax.plot([-1, 2], [0.0, 0.0], 'k')
ax.plot([4, 7], [0.0, 0.0], 'k')
ax.plot([2], [1 / 4.0], markersize = 5, marker='o', color='k')
ax.plot([2], [0.0], markersize = 5, marker='o',
        markeredgecolor='k', markeredgewidth=1, fillstyle='none')

ax.plot([4], [0.0], markersize = 5, marker='o', color='k')
ax.plot([4], [1 / 4.0], markersize = 5, marker='o',
        markeredgecolor='k', markeredgewidth=1, fillstyle='none')
# ax.plot([1], [0.], markersize = 5, marker='o',
#         markeredgecolor='k', markeredgewidth=1, fillstyle='none')
ax.set_title(r'Pdf and cdf of the r.v. $X$')

ax.plot([1, 2, 4, 7], [0.5, 0.5, 1.0, 1.0], 'r', label=r'$F_X$')
ax.plot([-1, 1], [0, 0], 'r')
# ax.plot([2], [0.5], markersize = 5, marker='o', color='r')
ax.plot([1], [0.], markersize = 5, marker='o', markeredgecolor='r', markeredgewidth=1, fillstyle='none')
ax.annotate('', xy=(1.0, 0.5), xytext=(1.0, 0.0),
            arrowprops={'arrowstyle': '-|>'}, va='center')

ax.legend()
plt.savefig('example_cdf_pdf.pgf')

# ----------------------------------------------------------------------
# Pdf of normal
plt.figure(figsize=col_full)
x = np.linspace(-5, 7, 200)
ax = plt.subplot(1, 2, 1)
ax.set_xlim([-5, 7.5])
ax.set_ylim([-0.02, 0.5])
ax.plot(x, scipy.stats.norm.pdf(x), label=r'$p_X,\, X\sim \mathcal{N}(0,1)$')
ax.plot(x, scipy.stats.norm.pdf(x, loc=1, scale=2), label=r'$p_X,\, X\sim \mathcal{N}(1,2)$')
ax.set_xlabel(r'$x$')
ax.legend()
ax.set_title(r'Pdf of two Gaussian r.v.')
ax = plt.subplot(1, 2, 2)
xmg, ymg = np.mgrid[-3:3:.01, -3:3:.01]
pos = np.dstack((xmg, ymg))
rv = scipy.stats.multivariate_normal([-1, -1], [[2.0, 0.3], [0.3, 0.5]])
ax.contour(xmg, ymg, rv.pdf(pos))
ax.set_title(r'Density of a 2D-Gaussian r.v.')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
plt.savefig('example_normal.pgf')




# EOF ----------------------------------------------------------------------


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import scipy.special

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


def Matern_fun(h, nu, rho=1):
    arg = np.sqrt(2 * nu) * h / rho
    return 2.0**(1 - nu) / scipy.special.gamma(nu) * scipy.special.kv(nu, arg) * (arg)**nu


def gaussian_fun(h, rho=1):
    return np.exp(- h**2 / (2 * rho**2))


xp = np.linspace(0, 3, 500)

plt.figure(figsize = col_full)
plt.plot(xp, Matern_fun(xp, 0.5), label=r'Exponential kernel')
plt.plot(xp, Matern_fun(xp, 3.0 / 2.0), label=r'Mat\'ern $3/2$')
plt.plot(xp, Matern_fun(xp, 5.0 / 2.0), label=r'Mat\'ern $5/2$')
plt.plot(xp, gaussian_fun(xp), label=r'Gaussian')
plt.legend()
plt.show()

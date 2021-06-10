#!/usr/bin/env python
# coding: utf-8
# ----------------------------------------------------------------
import numpy as np
import scipy
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

exec(open('/home/victor/acadwriting/Manuscrit/plots_settings.py').read())
# from pyDOE import *
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.serif": ["Computer Modern Roman"],
    'image.cmap': u'viridis',
    'figure.figsize': [6, 4],
    'savefig.dpi': 400
})
plt.rc('text.latex', preamble=r"\usepackage{amsmath} \usepackage{amssymb}")

Ndim = 2
Npts = 1000
NptsK, NptsU = Npts, Npts
xx = np.linspace(0, 1, Npts)
yy = xx.copy()
X_, Y_ = np.meshgrid(xx, yy, indexing = 'ij')
comb = np.array([X_, Y_]).T.reshape(-1, 2)


def branin_hoo(X, damp = 1.0, switch = False):
    y, x = X[:, 1], X[:, 0]
    if switch:
        x, y = y, x
    x2 = 15 * y
    x1 = 15 * x - 5

    quad = (x2 - (5.1 / (4 * np.pi**2)) * x1**2 + (5 / np.pi) * x1 - 6)**2
    cosi = (10 - (10 / np.pi * 8)) * np.cos(x1) - 44.81
    return (quad + cosi) / (51.95 * damp) + 2.0


def Jminus_min(alpha, y_fun, val_star_2d_true):
    value = y_fun - alpha * np.asarray(val_star_2d_true)[np.newaxis, :]
    bool_grid = (value <= 0.0)
    return value, bool_grid


def plot_original_function(ax, y_fun, resolution):
    ax.contourf(X_, Y_, y_fun, resolution)  # , cmap = cm.Greys)

    
def plot_conditional_minimisers(ax, x, yy, index=None):
    if index is None:
        ax.plot(x, yy, 'r.', markersize=3,
                label=r'$(\theta^*(u), u)$')
    else:
        ax.plot(x[index], yy[index], 'r.', markersize = 3)


def plot_alpha_regions(ax, X_, Y_, bool_grid):
    ax.contour(X_, Y_, bool_grid, levels=[0.5], colors='yellow')
    ax.plot(np.nan, np.nan, label=r'$\{J = \alpha J^*\}$', color='yellow')


def relaxation_tuto(alpha=1.5):
    funtest = lambda X: branin_hoo(X, switch=True)
    y_fun = funtest(comb).reshape(Npts, Npts)
    x_star_2d_true = np.asarray([xx[y_fun[:, uu].argmin()] for uu in range(NptsU)])
    val_star_2d_true = np.asarray([y_fun[:, uu].min() for uu in range(NptsU)])
    resolution = 10
    index = 280
    ind = 599
    ftsize = 12

    fig, axul = plt.subplots(figsize=(4, 4))
    axul.set_title(r'Conditional minimiser')
    plot_original_function(axul, y_fun, resolution)
    axul.axhline(yy[index], color = 'r', ls='--')
    axul.annotate(r'$u$',
                  xy=(0, yy[index]), xytext=(0.05, yy[index] + 0.05),
                  fontsize = ftsize, color = 'white')
    kstar = x_star_2d_true[index]
    axul.axvline(kstar, color = 'r', ls = '--')
    axul.annotate(r'$\theta^*(u)$',
                  xy=(kstar, 0),
                  xytext=(kstar + 0.05, 0.05), color='white',
                  fontsize = ftsize)
    axul.annotate(r'$J^*(u)$',
                  xy=(kstar, yy[index]),
                  xytext=(kstar + 0.1, yy[index] + 0.1), color='white',
                  arrowprops=dict(arrowstyle = '->', edgecolor='white'), fontsize = ftsize)
    plot_conditional_minimisers(axul, x_star_2d_true, yy, index)
    axul.set_xlabel(r'$\theta$')
    axul.set_ylabel(r'$u$')
    plt.savefig('/home/victor/acadwriting/Slides/Figures/relaxation_1.pgf')
    plt.close()

    fig, axll = plt.subplots(figsize=(4, 4))
    axll.set_title(r'Relaxation of the constraint, $\alpha={}$'.format(alpha))
    plot_original_function(axll, y_fun, resolution)

    value, bool_grid = Jminus_min(alpha, y_fun, val_star_2d_true)
    plot_alpha_regions(axll, X_, Y_, bool_grid)
    acceptable_values = np.where(bool_grid[:, ind], xx[ind], np.nan)
    bad_values = np.where(bool_grid[:, ind], np.nan, xx[ind])
    axll.annotate(r'$J(\theta,u) < \alpha J^*(u)$',
                  textcoords='data', xycoords='data', horizontalalignment='center',
                  xy = (1.7 / 5, 3 / 5),
                  xytext=(3.5 / 5, 4.5 / 5), color = 'cyan',
                  arrowprops=dict(arrowstyle='->', edgecolor='cyan',
                                  connectionstyle = 'angle, angleA=0, angleB=90',
                                  mutation_scale=16),
                  fontsize = ftsize)
    axll.annotate(r'$J(\theta,u) > \alpha J^*(u)$',
                  textcoords='data', xycoords='data', horizontalalignment='center',
                  xy = (3.5 / 5, 3 / 5),
                  xytext=(3.5 / 5, 4 / 5), color = 'magenta',
                  arrowprops=dict(arrowstyle='->', edgecolor='magenta',
                                  mutation_scale=16),
                  fontsize = ftsize)
    axll.plot(xx, acceptable_values, color = 'cyan')
    axll.plot(xx, bad_values, color = 'magenta')
    plot_conditional_minimisers(axll, x_star_2d_true, yy)

    leg = axll.legend(loc='lower left', fontsize=8)
    for text in leg.get_texts():
        plt.setp(text, color = 'w')
    
    axll.axvline(xx[300], color='white', ls='--')
    # axll.annotate(r'$\theta$',
    #               textcoords='data', xycoords='data', color='white',
    #               fontsize = ftsize,
    #               xy = (xx[300], 0), xytext=(xx[300] - 0.05, 0.05))
    axll.set_xlabel(r'$\theta$')
    axll.set_ylabel(r'$u$')
    plt.savefig('/home/victor/acadwriting/Slides/Figures/relaxation_3.pgf')
    plt.close()

    # --- Second column -----------------------------------------------------

    fig, axur = plt.subplots(figsize=(4, 4))
    axur.set_title(r'Set of conditional minimisers')
    plot_original_function(axur, y_fun, resolution)
    plot_conditional_minimisers(axur, x_star_2d_true, yy)
    leg = axur.legend(loc='lower left', fontsize=8)
    for text in leg.get_texts():
        plt.setp(text, color = 'w')
    axur.set_xlabel(r'$\theta$')
    axur.set_ylabel(r'$u$')
    plt.savefig('/home/victor/acadwriting/Slides/Figures/relaxation_2.pgf')
    plt.close()


    fig, axlr = plt.subplots(figsize=(4, 4))
    axlr.set_title(r'Region $R_{\alpha}(\theta)$')
    plot_original_function(axlr, y_fun, resolution)
    plot_alpha_regions(axlr, X_, Y_, bool_grid)
    plot_conditional_minimisers(axlr, x_star_2d_true, yy)

    u_ind = 300  # ((x_star_2d_true - xx)**2).argmin()
    axlr.axvline(xx[300], color='white', ls='--')
    plot_conditional_minimisers(axur, x_star_2d_true, yy)

    # axlr.annotate(r'$\theta$',
    #               textcoords='data', xycoords='data', color='white',
    #               fontsize = ftsize,
    #               xy = (xx[300], 0), xytext=(xx[300] - 0.05, 0.05))
    values_prob = np.where(bool_grid[u_ind, :], xx[u_ind] * bool_grid[u_ind, :], np.nan)
    axlr.plot(values_prob, yy, color='lime', markersize = 1)
    mid = np.nanmean(np.where(bool_grid[u_ind, :], yy * bool_grid[u_ind, :], np.nan))
    rge = np.nanmax(np.where(bool_grid[u_ind, :], yy * bool_grid[u_ind, :], np.nan))\
        - np.nanmin(np.where(bool_grid[u_ind, :], yy * bool_grid[u_ind, :], np.nan))
    axlr.annotate(r'$R_{\alpha}(\theta)$',
                  xy=(xx[u_ind] + 0.2, mid),
                  xytext=(3.5 / 5., 3 / 5), color='lime',
                  textcoords='data', xycoords='data', fontsize=ftsize,
                  arrowprops=dict(arrowstyle='-[, widthB=' + str(rge + 6.0),
                                  edgecolor='lime',
                                  connectionstyle = 'angle, angleA=90, angleB=0',
                                  mutation_scale=ftsize + 1)
                  )

    leg = axlr.legend(loc='upper right', fontsize=8)
    for text in leg.get_texts():
        plt.setp(text, color = 'w')
    
    axlr.set_xlabel(r'$\theta$')
    axlr.set_ylabel(r'$u$')

    plt.savefig('/home/victor/acadwriting/Slides/Figures/relaxation_4.pgf')
    plt.close()


def plots_different_alpha(alpha1=1.05, p2=.90):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, figsize=(6, 6))

    # top plots
    funtest = lambda X: branin_hoo(X, switch=True)
    y_fun = funtest(comb).reshape(Npts, Npts)
    x_star_2d_true = np.asarray([xx[y_fun[:, uu].argmin()] for uu in range(NptsU)])
    val_star_2d_true = np.asarray([y_fun[:, uu].min() for uu in range(NptsU)])
    resolution = 10
    plot_original_function(ax1, y_fun, resolution)
    plot_conditional_minimisers(ax1, x_star_2d_true, yy, None)
    plot_original_function(ax2, y_fun, resolution)
    plot_conditional_minimisers(ax2, x_star_2d_true, yy, None)
    leg1 = ax1.legend(loc='upper right')
    leg2 = ax2.legend(loc='upper right')
    for text in leg1.get_texts():
        plt.setp(text, color = 'w')
    for text in leg2.get_texts():
        plt.setp(text, color = 'w')
    
    value1, bool_grid1 = Jminus_min(alpha1, y_fun, val_star_2d_true)
    alpha2 = np.quantile(y_fun / val_star_2d_true, p2, 1).min()
    value2, bool_grid2 = Jminus_min(alpha2, y_fun, val_star_2d_true)

    # acceptable_values = np.where(bool_grid1[:, ind], xx[ind], np.nan)
    plot_alpha_regions(ax1, X_, Y_, bool_grid1)
    plot_alpha_regions(ax2, X_, Y_, bool_grid2)

    # bottom plots
    ax3.plot(xx, bool_grid1.mean(1))
    ax3.axhline(bool_grid1.mean(1).max(), color='grey', ls='--')
    ax4.plot(xx, bool_grid2.mean(1))
    ax4.axhline(p2, color='grey', ls='--')

    u_ind = bool_grid1.mean(1).argmax()
    values_prob = np.where(bool_grid1[u_ind, :], xx[u_ind] * bool_grid1[u_ind, :], np.nan)
    ax1.plot(values_prob, yy, color='lime', markersize = 1)
    
    u_ind = bool_grid2.mean(1).argmax()
    values_prob = np.where(bool_grid2[u_ind, :], xx[u_ind] * bool_grid2[u_ind, :], np.nan)
    ax2.plot(values_prob, yy, color='lime', markersize = 1)
    ax3.set_ylim([0, 1])
    ax4.set_ylim([0, 1])

    ax1.set_xlabel(r'$\theta$')
    ax2.set_xlabel(r'$\theta$')
    ax3.set_xlabel(r'$\theta$')
    ax4.set_xlabel(r'$\theta$')
    ax1.set_ylabel(r'$u$')
    ax2.set_ylabel(r'$u$')
    ax3.set_ylabel(r'$\Gamma_{\alpha}$')
    ax4.set_ylabel(r'$\Gamma_{\alpha}$')

    ax1.set_title(r'Region of acceptability for $\alpha={:.2f}$'.format(alpha1))
    ax2.set_title(r'Region of acceptability for $\alpha={:.2f}$'.format(alpha2))

    ax3.set_title(r'$\Gamma_\alpha, \max \Gamma_\alpha = {:.2f}$'.format(bool_grid1.mean(1).max()))
    ax4.set_title(r'$\Gamma_\alpha, \max \Gamma_\alpha = {:.2f}$'.format(p2))
    plt.tight_layout()
    plt.savefig(r'/home/victor/acadwriting/Slides/Figures/relaxation_2sides.pgf')
    plt.close()

if __name__ == '__main__':
    relaxation_tuto()
    plots_different_alpha()

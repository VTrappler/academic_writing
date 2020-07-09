#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
import numpy as np
import scipy.stats
import scipy.special
# import RO.bo_plot as bplt
# import RO.bo_wrapper as bow
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

import pyDOE

exec(open('/home/victor/acadwriting/Manuscrit/plots_settings.py').read())


def plot_gp(gp, X_, true_function=None, nsamples=0, show=True, label=None):
    """
    Plot a 1D Gaussian Process, with CI and samples
    """
    y_mean, y_std = gp.predict(X_[:, np.newaxis], return_std=True)
    y_mean = y_mean.squeeze()
    if true_function is not None:
        true, = plt.plot(X_, true_function(X_), 'r--', label=label)

    reg, = plt.plot(X_, y_mean, 'k', lw=1, zorder=9, label=r'GP regression')
    sh1 = plt.fill_between(X_,
                     y_mean - y_std,
                     y_mean + y_std,
                     alpha=0.1, color='k', linewidth=0.0)
    sh2 = plt.fill_between(X_,
                     y_mean - 2 * y_std,
                     y_mean + 2 * y_std,
                     alpha=0.05, color='k', linewidth=0.0)
    sh3 = plt.fill_between(X_,
                     y_mean - 3 * y_std,
                     y_mean + 3 * y_std,
                     alpha=0.05, color='k', linewidth=0.0)

    if nsamples > 0:
        y_samples = gp.sample_y(X_[:, np.newaxis], 10)
        plt.plot(X_, y_samples, lw=1)

    plt.plot(gp.X_train_, gp.y_train_, 'ob', label=r'Training points')
    return true, reg, sh1, sh2, sh3


def Matern_fun(h, nu, rho=1):
    arg = np.sqrt(2 * nu) * h / rho
    return 2.0**(1 - nu) / scipy.special.gamma(nu) * scipy.special.kv(nu, arg) * (arg)**nu


def gaussian_fun(h, rho=1):
    return np.exp(- h**2 / (2 * rho**2))

function_gp = lambda X: X * np.sin(X)


np.random.seed(3367)
bounds = [0, 2 * np.pi]
X_ = np.linspace(*bounds, num=200)
initial_design = pyDOE.lhs(n=1,
                           samples=5,
                           criterion='maximin',
                           iterations=50) * (bounds[1] - bounds[0]) + bounds[0]
response = function_gp(initial_design)

    # Fitting of the GaussianProcess -------------------------------------
gp = GaussianProcessRegressor(kernel=Matern(1.0 / 5.0),
                              n_restarts_optimizer=50)
gp.fit(initial_design, response)
true, reg, sh1, sh2, sh3 = plot_gp(gp, X_, true_function=function_gp, show=False, label=r'True function $f$')
plt.legend([true, reg, (sh1, sh2, sh3)], handler_map={tuple: HandlerTuple(ndivide=None)})
plt.xlabel(r'$x$')
plt.xlim(bounds)
plt.show()



plt.figure(figsize = col_full)
plt.subplot(1, 2, 1)
plt.plot(xp, Matern_fun(xp, 0.5), label=r'Exponential kernel')
plt.plot(xp, Matern_fun(xp, 3.0 / 2.0), label=r'Mat\'ern $3/2$')
plt.plot(xp, Matern_fun(xp, 5.0 / 2.0), label=r'Mat\'ern $5/2$')
plt.plot(xp, gaussian_fun(xp), label=r'Gaussian')
plt.legend()
plt.xlabel(r'$h$')
plt.ylabel(r'$Cov(x, x+h)$')
plt.title(u'Common covariance function for GP regression')
plt.tight_layout()

plt.subplot(1, 2, 2)


plt.savefig('./img/covariance_functions.pgf')
plt.close()
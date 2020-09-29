#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
import numpy as np
import scipy.stats
import scipy.special
import sys
# import RO.bo_plot as bplt
sys.path.append('/home/victor/RO_VT/RO/')

import RO.bo_wrapper as bow
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF

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

plt.figure(figsize=col_full)
true, reg, sh1, sh2, sh3 = plot_gp(gp, X_, true_function=function_gp, show=False, label=r'True function $f$')
# plt.plot(np.nan, 'ob', label=r'Training points $f(x_i)$')
# plt.plot(np.nan, label=r'GP regression', color='k', lw=1)
# plt.plot(np.nan, 'r--', label=r'True function $f$')
# plt.legend([true, reg, (sh1, sh2, sh3)], handler_map={tuple: HandlerTuple(ndivide=None)})
plt.legend()
plt.xlabel(r'$x$')
plt.xlim(bounds)
plt.savefig('./img/example_GP.pgf')
plt.close()

xp = np.linspace(0, 4, 200)
plt.figure(figsize=(col_full[0], col_full[1] * 1.2))
plt.subplot(4, 2, (1, 7))
plt.plot(xp, Matern_fun(xp, 0.5), label=r'Exponential')
plt.plot(xp, Matern_fun(xp, 3.0 / 2.0), label=r'Mat\'ern $3/2$')
plt.plot(xp, Matern_fun(xp, 5.0 / 2.0), label=r'Mat\'ern $5/2$')
plt.plot(xp, gaussian_fun(xp), label=r'Gaussian')
plt.legend()
plt.xlabel(r'$h$')
plt.ylabel(r'$C_Z(h)$')
plt.title(u'Common covariance functions')

nsamples=10
kernels = [Matern(1.0, nu=0.5), Matern(1.0, nu=1.5), Matern(1.0, nu=3.5), RBF(1.0)]
lab = [r'Exponential', r'Mat\'ern 3/2', r'Mat\'ern 5/2', r'Gaussian']
for i, ker in enumerate(kernels):
    plt.subplot(4, 2, (i + 1) * 2)
    gp = GaussianProcessRegressor(kernel=ker)
    plt.plot(gp.sample_y(xp[:, np.newaxis], nsamples).squeeze(), linewidth=1)
    plt.title(lab[i])
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.savefig('./img/covariance_functions.pgf')
plt.close()


plt.figure(figsize=col_full)
plt.subplot(2, 1, 1)
true, reg, sh1, sh2, sh3 = plot_gp(gp, X_,
                                   true_function=function_gp,
                                   show=False,
                                   label=r'True function $f$')
plt.legend()
plt.xlabel(r'$x$')
plt.xlim(bounds)

ax = plt.subplot(2, 1, 2)
m_Z, s_Z = gp.predict(X_[:, np.newaxis], return_std=True)
ax.plot(X_, s_Z**2, label=r'$\sigma_Z$')
IMSE = []
scenarios = lambda mp, sp: scipy.stats.norm.ppf(np.linspace(0.05, 0.95, 10, endpoint=True),
                                                loc=mp, scale=sp)
for x in X_:
    mp, sp = gp.predict(np.atleast_2d(x), return_std=True)
    evaluated_points = scenarios(mp, sp).squeeze()
    ss = 0
    for evalu in evaluated_points:
        gpp = bow.add_points_to_design(gp, x, evalu)
        ss += bow.integrated_variance(gpp, X_[:, np.newaxis], alpha=None)
    IMSE.append(ss / float(len(evaluated_points)))
ax2 = ax.twinx()
ax2.plot(X_, -np.asarray(IMSE), label=r'$-$ augmented IMSE', color=colors[1])
ax2.plot(np.nan, np.nan, label=r'$\sigma_Z$', color=colors[0])
ax.legend()
plt.show()
plt.savefig('./img/IMSE_variance.pgf')



aIVPC_2stage_UB
aIVPC = np.load('/home/victor/Bureau/aIVPC_new.npy', allow_pickle=True)

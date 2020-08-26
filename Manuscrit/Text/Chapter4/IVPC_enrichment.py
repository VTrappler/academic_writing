#!/usr/bin/env python3
# coding: utf-8

from __future__ import unicode_literals, print_function, with_statement
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from matplotlib import pyplot as plt
import itertools
from sklearn.gaussian_process.kernels import Matern
from sklearn.cluster import KMeans
import scipy
import pyDOE
import copy
import warnings
import time
import sys

sys.path.append('/home/victor/RO_VT/')
import RO.bo_wrapper as bow

import matplotlib as mpl
from matplotlib import cm
plt.style.use('seaborn')
mpl.rcParams['image.cmap'] = u'viridis'
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
exec(open('/home/victor/acadwriting/Manuscrit/plots_settings.py').read())




def rm_obs_gp(gp, ninitial, n_added):
    gp_rm = copy.copy(gp)
    N = ninitial + n_added
    X_train_ = gp.X_train_[:N]
    y_train_ = gp.y_train_[:N]
    gp_rm.fit(X_train_, y_train_)
    return gp_rm




if __name__== '__main__':

    
    from RO.test_functions import branin_2d
    function_2d = lambda X: branin_2d(X, switch=False)
    rng = np.random.RandomState(3394)
    ndim = 2
    bounds = np.asarray([[0, 1], [0, 1]])

    # Builds a regular grid ---------------------------------------------
    ngrid = 50
    X_, Y_ = np.linspace(0, 1, ngrid), np.linspace(0, 1, ngrid)
    # xx, yy = np.meshgrid(X_, Y_, indexing = 'ij')
    all_combinations, (X_mg, Y_mg) = bow.pairify(X_, Y_)

    ngrid_big, ngrid_big_2 = 1000, 2000
    X_l, X_l2 = np.linspace(0, 1, ngrid_big), np.linspace(0, 1, ngrid_big_2)
    big_comb, (mg_b1, mg_b2) = bow.pairify(X_l, X_l2)


    p = .95
    out_t = function_2d(big_comb).reshape(ngrid_big, ngrid_big_2)
    kstar_t = out_t.argmin(0)
    Jstar_t = out_t.min(0)
   
    rho_t = (out_t / Jstar_t[np.newaxis, :])
    alpha_t = np.quantile(rho_t, p, axis=1)


    out_t_allc = function_2d(all_combinations).reshape(ngrid, ngrid)
    kstar_t_allc = out_t_allc.argmin(0)
    Jstar_t_allc = out_t_allc.min(0)
    delta_t_allc = out_t_allc - 1.8 * Jstar_t_allc[np.newaxis, :] <= 0
    
    aIVPC = np.load('/home/victor/Bureau/aIVPC.npy', allow_pickle=True)
    gp_aIVPC = GaussianProcessRegressor(kernel=Matern(np.ones(ndim) / 5.0),
                                            n_restarts_optimizer=50)
    gp_aIVPC.fit(aIVPC[()]['X_train_'], aIVPC[()]['y_train_'])
    
    gp = GaussianProcessRegressor(kernel=Matern(np.ones(ndim) / 5.0),
                                  n_restarts_optimizer=50)
    gp.fit(aIVPC[()]['X_train_'][:30], aIVPC[()]['y_train_'][:30])

    IVPC = aIVPC[()]['IVPC']
    plugin = aIVPC[()]['plugin']
    ppi = aIVPC[()]['ppi']
    
    # VPC_after = variance_of_prob_coverage(gp_aIVPC, 1.8, all_combinations)
    # VPC_before = variance_of_prob_coverage(gp, 1.8, all_combinations)
    # bplt.plot_2d_strategy(gp_aIVPC, all_combinations, function_2d,
    #                       VPC_after)
    # bplt.plot_2d_strategy(gp, all_combinations, function_2d,
    #                       VPC_before)


    delta = out_t - 1.8 * Jstar_t[np.newaxis, :] <= 0
    
    L2_norm_PI = []
    L2_norm_PC = []
    Linf_norm_PI = []
    Linf_norm_PC = []
    ptarget = delta.mean(1).max()
    dist_to_ptarget_PI = []
    dist_to_ptarget_PC = []
    for i in range(71):
        pp = ppi[i]
        gamma_PI = (plugin[i] <= 0).mean(1)
        gamma_PC = pp.mean(1)
        plt.plot(X_, gamma_PI, alpha=0.1, color='b')
        plt.plot(X_, gamma_PC, alpha=0.1, color='g')
        
        L2_norm_PI.append(np.sum(gamma_PI - delta_t_allc.mean(1))**2)
        L2_norm_PC.append(np.sum(gamma_PC - delta_t_allc.mean(1))**2)
        
        Linf_norm_PI.append(np.abs(gamma_PI - delta_t_allc.mean(1)).max())
        Linf_norm_PC.append(np.abs(gamma_PC - delta_t_allc.mean(1)).max())

        dist_to_ptarget_PI.append(np.abs(gamma_PI.max() - ptarget))
        dist_to_ptarget_PC.append(np.abs(gamma_PC.max() - ptarget))
    plt.plot(X_l, delta.mean(1), color='r')
    plt.show()



    plt.figure(figsize=col_full)
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(L2_norm_PI, label='PI')
    ax1.plot(L2_norm_PC, label='$\pi$')
    ax1.legend()
    ax1.set_title(r'$\|\hat{\Gamma}_{\alpha,n} - \Gamma_{\alpha}\|_2$')
    ax1.set_ylabel(r'$L^2$ norm')

    ax1.set_yscale('log')
    ax2 = plt.subplot(2, 2, 3)
    ax2.plot(Linf_norm_PI, label='PI')
    ax2.plot(Linf_norm_PC, label='$\pi$')
    ax2.set_title(r'$\|\hat{\Gamma}_{\alpha,n} - \Gamma_{\alpha}\|_{\infty}$')
    ax2.set_ylabel(r'$L^{\infty}$ norm')

    ax2.legend()
    ax2.set_yscale('log')
    # ax3.plot(dist_to_ptarget_PI, label='PI')
    # ax3.plot(dist_to_ptarget_PC, label='$\pi$')
    # ax3.set_yscale('log')
    # ax3.set_title(r'$|p_{\alpha} - \hat{p}_{\alpha}|$')
    # ax3.legend()
    ax4 = plt.subplot(2, 2, (2, 4))
    ax4.plot(IVPC)
    # ax4.set_yscale('log')
    ax4.set_title(r'IVPC($\mathcal{X}_n$)')
    ax4.set_ylabel(r'IVPC')
    for ax in [ax1, ax2, ax4]:
        ax.set_xlabel(r'$n$')
    plt.tight_layout()
    plt.savefig('/home/victor/acadwriting/Manuscrit/Text/Chapter4/img/IVPC_enrichment.pgf')
    plt.close()

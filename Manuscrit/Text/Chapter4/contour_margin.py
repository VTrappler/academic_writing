#!/usr/bin/env python3
# coding: utf-8

from __future__ import unicode_literals, print_function, with_statement
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from matplotlib import pyplot as plt
import matplotlib as mpl
import itertools
from sklearn.gaussian_process.kernels import Matern
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import scipy
import pyDOE
import copy
import warnings
import time
import sys
import matplotlib as mpl
from matplotlib import cm


sys.path.append('/home/victor/RO_VT/')
import RO.bo_wrapper as bow
from RO.test_functions import branin_2d


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)





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

print(sys.version)
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=b"\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts}")
plt.style.use('seaborn')
plt.rc('font', **{'family': 'serif',
                  'serif': ['Computer Modern Roman']})
plt.rcParams.update(params)

col_half = get_figsize()
col_full = get_figsize(wf=1.0)
col_3quarter = get_figsize(wf=.75)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

mpl.rcParams['image.cmap'] = u'viridis'

# exec(open('/home/victor/acadwriting/Manuscrit/plots_settings.py').read())



def adjust_centroid(gp, centro, alpha):
    m2, cov2 = bow.mean_covariance_alpha(gp,
                                         np.atleast_2d(centro[0]),
                                         np.atleast_2d(centro[1]), [1], np.asarray([[0, 1]]))

    # _, ss = mu_sigma_delta(gp, np.atleast_2d(centro), alpha, [1], np.asarray([0, 1]))

    print('Adjust the centroid: ', cov2[0, 0] <= alpha**2 * cov2[1, 1])
    if cov2[0, 0] <= alpha**2 * cov2[1, 1]:
        curr_min = bow.find_minimum_sliced(gp, centro[1], [1], bounds=np.asarray([[0, 1]]))
        kstar = curr_min.x[0]
        f_min = curr_min.fun
        sliced_fun = bow.slicer_gp_predict(gp, np.asarray(centro[1]), [1], return_std=True)

        def EI_sliced(X_):
            y_mean, y_std = sliced_fun(np.atleast_2d(X_))
            m = f_min - y_mean
            return -bow.acq.expected_improvement_closed_form(m, y_std)
        i = 0
        minval = np.inf
        while i < 5:
            opt = scipy.optimize.minimize(EI_sliced, np.random.uniform(),
                                          bounds=np.atleast_2d([0, 1]))
            if opt.fun < minval:
                curr = opt
                minval = curr.fun
            i += 1
            kEI = curr.x[0]
        print('kstar:', kstar)
        print('kEI:', kEI)
        newku = kstar, centro[1]
    else:
        newku = centro
    return np.asarray(newku), (cov2[0, 0], alpha**2 * cov2[1, 1])


def main():
    function_2d = lambda X: branin_2d(X, switch=False)
    np.random.seed(3394)
    ndim = 2
    bounds = np.asarray([[0, 1], [0, 1]])
    initial_design = pyDOE.lhs(2, 30, criterion='maximin', iterations=50)
    # Builds a regular grid ---------------------------------------------
    ngrid = 120
    X_, Y_ = np.linspace(0, 1, ngrid), np.linspace(0, 1, ngrid)
    all_combinations, (X_mg, Y_mg) = bow.pairify(X_, Y_)

    # Threshold = 1.8
    out = function_2d(all_combinations).reshape(ngrid, ngrid)
    kstar = out.argmin(0)
    Jstar = out.min(0)
    delta = out - 1.8 * Jstar[np.newaxis, :] <= 0

    gp = bow.GaussianProcessRegressor(kernel=Matern(np.ones(ndim) / 5.0),
                                      n_restarts_optimizer=50)
    gp.fit(initial_design, function_2d(initial_design))

    def margin_indicator_delta(gp, x, eta=0.025):
        return bow.margin_indicator(bow.mu_sigma_delta(gp,
                                                       x,
                                                       1.8,
                                                       [1],
                                                       np.asarray([0, 1]),
                                                       verbose=False),
                                    0, 1 - eta, x)

    m, s = bow.mu_sigma_delta(gp, all_combinations, 1.8, [1], np.asarray([0, 1]), verbose=True)

    plt.figure(figsize=(col_full[0] * 0.95, col_full[1] * 1.3))
    cmap = cm.get_cmap('Pastel2', 4)
    plt.subplot(2, 2, 1)
    # gpkstar = gp.predict(all_combinations).reshape(ngrid, ngrid).argmin(0)
    plt.contourf(X_mg, Y_mg, gp.predict(all_combinations).reshape(ngrid, ngrid), 10)
    plt.plot(gp.X_train_[:, 0], gp.X_train_[:, 1], '.', color='red')
    # plt.plot(X_[gpkstar], X_, 'white', '.')
    # plt.plot(X_[kstar], X_, 'blue', '.')
    plt.title(r'GP prediction')
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$u$')
    plt.subplot(2, 2, 2)
    cov_prob = bow.coverage_probability((m, s), 0, all_combinations)
    plt.contourf(X_mg, Y_mg, cov_prob.reshape(ngrid, ngrid), 10)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$u$')
    cbar = plt.colorbar()
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([0, 1])
    cbar.set_label(r'$\pi_\alpha$', rotation=0)
    plt.title(r'Coverage probability $\pi_{\alpha}$' + '\n' + r'for $\{\Delta_{\alpha} \leq 0\}$')
    plt.subplot(2, 2, 4)
    margin_of_uncertainty = bow.margin_indicator((m, s), 0, 0.975, all_combinations)
    print('size of Meta: {}'.format(np.mean(margin_of_uncertainty)))
    CS = plt.contourf(X_mg, Y_mg, margin_of_uncertainty.reshape(ngrid, ngrid),
                      levels=[-0.1, 0.5, 1.1],
                      cmap=cmap)
    cbar = plt.colorbar(boundaries=[0, 0.5, 1])
    # cbar.set_label(r'$\mathbb{M}_{\eta}$', rotation=0)
    # cbar.set_ticks([0, 1])
    tick_locs = (np.arange(2) + 0.5) / 2
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels([r'$\mathbb{M}_{\eta}^C$', r'$\mathbb{M}_{\eta}$'])
    # plt.clabel(CS)
    # plt.annotate(r'$\mathbb{M}_{\eta}$', xy=(0.7, 0.5))
    # plt.annotate(r'$\mathbb{M}_{\eta}^C$', xy=(0.6, 0.8), color='white')
    plt.title(r'Margin of uncertainty' + '\n' + r'for the GP $\Delta_{\alpha}$')
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$u$')
    plt.subplot(2, 2, 3)
    plt.plot(X_, (cov_prob * (1 - cov_prob)).reshape(ngrid, ngrid).mean(1))
    plt.title(r'IVPC computed at constant $\theta$')
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'IVPC')
    plt.tight_layout()
    plt.savefig('/home/victor/acadwriting/Manuscrit/Text/Chapter4/img/prob_coverage_margin.pdf')
    plt.show()



    gp_tmp = gp
    cmap = cm.get_cmap('Pastel2', 4)
    plt.figure(figsize=(col_full[0] * 1.3, col_full[1]*1))
    for j in range(2):
        m, s = bow.mu_sigma_delta(gp_tmp, all_combinations, 1.8, [1],
                                  np.asarray([0, 1]), verbose=True)
        cov_prob = bow.coverage_probability((m, s), 0, all_combinations)
        margin_of_uncertainty = bow.margin_indicator((m, s), 0, 0.975, all_combinations)
        print('size of Meta: {}'.format(np.mean(margin_of_uncertainty)))

        samples_margin = bow.sample_from_criterion(1000,
                                                   lambda x: margin_indicator_delta(gp_tmp, x),
                                                   bounds=np.asarray([[0, 1],
                                                                      [0, 1]]),
                                                   Ncandidates=3)
        kmeans = bow.cluster_and_find_closest(10, samples_margin)
        opt = {'head_width': 0.02, 'head_length': 0.01, 'width': 0.001,
               'length_includes_head': True, 'alpha': 0.7, 'color': 'black'}
        plt.subplot(1, 2, j + 1)
        CS = plt.contour(X_mg, Y_mg, cov_prob.reshape(ngrid, ngrid),
                         levels=[0.025, 0.5, 0.975], cmap=cm.get_cmap('Dark2'))
        plt.clabel(CS, inline=True, fontsize=10, fmt=r'$\pi_{\alpha}$=%1.3f')
        plt.scatter(samples_margin[:, 0], samples_margin[:, 1], marker='.', c='grey', s=3,
                    label=r'Samples')
        plt.plot(kmeans[0][:, 0], kmeans[0][:, 1], 'red', marker='*', linestyle='',
                 label=r'Centroids')
        kadj = np.empty_like(kmeans[0])
        var_list = []
        for i, km in enumerate(kmeans[0]):
            kadj[i, :], (s2, alp_s2star) = adjust_centroid(gp_tmp, km, 1.8)
            var_list.append((s2, alp_s2star))
            if np.any(kadj[i, :] != km):
                dx, dy = kadj[i, :] - km
                # plt.arrow(km[0], km[1], dx, dy, **opt)

        var_list = np.asarray(var_list)
        kadj2 = np.copy(kmeans[0])
        hier_clusters = scipy.cluster.hierarchy.fclusterdata(kadj, 0.3)
        for cluster_index in np.unique(hier_clusters):
            cl = np.asarray(hier_clusters == cluster_index).nonzero()
            print(len(cl[0]))
            to_adjust = var_list[cl][:, 0].argmin()
            kadj2[cl[0][to_adjust]] = kadj[cl[0][to_adjust]]


        # scipy.cluster.hierarchy.dendrogram(scipy.cluster.hierarchy.linkage(kadj));plt.show()
        # plt.plot(gp.X_train_[:, 0], gp.X_train_[:, 1], '.', color='red', label='Evaluated points')
        plt.plot(kadj[:, 0], kadj[:, 1], color='blue', marker='*',
                 linestyle='', label=r'Adjusted centroids')
        plt.plot(kadj2[:, 0], kadj2[:, 1], color='magenta', marker='^',
                 linestyle='', label=r'Adjusted centroids 2')
        plt.ylabel(r'$u$')
        plt.xlabel(r'$\theta$')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        gp_tmp = bow.add_points_to_design(gp_tmp, kadj2, function_2d(kadj2), optimize_cov=True)
        gp_noadjustment = bow.add_points_to_design(gp_tmp, kmeans[0], function_2d(kmeans[0]),
                                                   optimize_cov=True)
        mna, sna = bow.mu_sigma_delta(gp_noadjustment, all_combinations, 1.8, [1],
                                      np.asarray([0, 1]), verbose=True)
        if j == 0:
            # plt.legend(fontsize=8)
            plt.title(r'Samples in $\mathbb{M}_{\eta}$' + '\nand adjusted centroids')
        else:
            plt.legend(fontsize=8)
            plt.title(r'Samples in $\mathbb{M}_{\eta}$' + '\nand adjusted centroids, after the evaluations')

        print('size of Meta, no adjustments: {}'.
              format(np.mean(bow.margin_indicator((mna, sna), 0, 0.975, all_combinations))))

    plt.tight_layout()
    plt.savefig('/home/victor/acadwriting/Manuscrit/Text/Chapter4/img/adjusted_centroids.pdf')
    plt.show()


if __name__ == '__main__':
    main()

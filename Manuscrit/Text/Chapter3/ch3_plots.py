#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import matplotlib as mpl
from matplotlib import patches
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.colors import ListedColormap

import numpy as np
import scipy.stats
exec(open('/home/victor/acadwriting/Manuscrit/plots_settings.py').read())
# # -> Manuscrit 415.41025
# # -> Notes 418.25368

# def get_figsize(columnwidth=415.41025, wf=0.5, hf=(5.**0.5 - 1.0) / 2.0):
#     """Parameters:
#       - wf [float]:  width fraction in columnwidth units
#       - hf [float]:  height fraction in columnwidth units.
#                      Set by default to golden ratio.
#       - columnwidth [float]: width of the column in latex. Get this from LaTeX 
#                              using \showthe\columnwidth
#     Returns:  [fig_width,fig_height]: that should be given to matplotlib
#     """
#     fig_width_pt = columnwidth * wf
#     inches_per_pt = 1.0 / 72.27               # Convert pt to inch
#     fig_width = fig_width_pt * inches_per_pt  # width in inches
#     fig_height = fig_width * hf      # height in inches
#     return [fig_width, fig_height]


# params = {'backend': 'pgf',
#           'axes.labelsize': 10,
#           'axes.titlesize': 11,
#           'image.cmap': u'viridis'}  # extend as needed


# mpl.use('pgf')
# plt.style.use('seaborn')
# plt.rc('font', **{'family': 'serif',
#                   'serif': ['Computer Modern Roman']})
# plt.rcParams.update(params)
# plt.rc('text', usetex=True)
# col_half = get_figsize()
# col_full = get_figsize(wf=1.0)


## Banana
plt.figure(figsize=col_full)
np.random.seed(3394)
y = scipy.stats.norm.rvs(size=10, loc=1, scale=2)

@np.vectorize
def lik(k, u, sig=2):
    return np.prod(np.exp(-(k + u**2 - y)**2 / (2 * sig**2)) / (np.sqrt(2 * np.pi) * sig))


def prior(k, u):
    return scipy.stats.norm.pdf(k, loc=0, scale=1)# * scipy.stats.norm.pdf(u, loc=0, scale=1)


kk, uu = np.linspace(-5, 5, 500), np.linspace(-2, 2, 400)
dk = kk[1] - kk[0]
kmg, umg = np.meshgrid(kk, uu)
postkmg = lik(kmg, umg) * prior(kmg, umg)
likmg = lik(kmg, umg)
plt.subplot(2, 2, 1)
plt.contourf(kk, uu, likmg)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$u$')
plt.title(r'Likelihood: $p_{Y \mid \theta, U}$')
plt.subplot(2, 2, 3)
plt.contourf(kk, uu, postkmg)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$u$')
plt.title(r'Posterior: $p_{\theta, U \mid Y}$')
ax = plt.subplot(2, 2, (2, 4))
ax.plot(kk, (likmg.mean(0)), label=r'$\mathcal{L}_{\mathrm{integrated}}$')
ax.plot(kk, (likmg.max(0)), label=r'$\mathcal{L}_{\mathrm{profile}}$')
ax.plot(kk, (postkmg.mean(0)), label=r'$p_{\theta \mid Y}$')
plt.vlines(kk[likmg.mean(0).argmax()],
         ymin=0, ymax = likmg.mean(0).max(),
         label=r'$\hat{\theta}_{\mathrm{intLik}}$', color=colors[0], ls=':')
plt.vlines(kk[postkmg.mean(0).argmax()],
         ymin=0, ymax = postkmg.mean(0).max(),
         label=r'$\hat{\theta}_{\mathrm{MMAP}}$', color=colors[2], ls=':')
ax.set_xlabel(r'$\theta$')
ax.set_xlim([-5, 5])
ax.set_yticks([0.])
ax.set_title(u'Profile, integrated likelihood,\nMarginal Posterior')
# ax.yaxis.set_major_formatter(plt.LogFormatter(10, labelOnlyBase=False))
ax.ticklabel_format(axis='y', style='sci', useMathText=False)
# plt.plot(postkmg.mean(0), label=r'Integrated Posterior')
# plt.plot(postkmg.max(0), label=r'Profile Posterior')
plt.legend()
plt.tight_layout()
plt.savefig('./img/profile_integrated_lik.pgf')
plt.close()


k, u = np.linspace(-5, 5, 500), np.linspace(-2, 2, 400)
kmg, umg = np.meshgrid(k, u)
postkmg = lik(kmg, umg) * prior(kmg, umg)
likmg = lik(kmg, umg)
plt.subplot(2, 2, 1)
plt.contourf(k, u, likmg)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$u$')
plt.title(r'Likelihood: $p_{Y \mid \theta, U}$')
ax = plt.subplot(2, 2, 2)
plt.contourf(k, u, postkmg)
plt.subplot(2, 2, 3)
plt.plot(k, likmg.mean(0))
plt.subplot(2, 2, 4)
plt.plot(k, postkmg.mean(0))
# plt.show()



plt.figure(figsize= [0.7 * col_full[0], col_full[1]])
# plt.subplot(1, 2, 1)
# plt.plot(kk, likmg.mean(0), label=r'$\mathcal{L}_{\mathrm{integrated}}(\theta;y)$')
# plt.plot(kk, np.exp(np.log(likmg).mean(0)), label=r'$\exp(-\mu(\theta))$')
# plt.axvline(kk[likmg.mean(0).argmax()], label=r'$\theta_{\mathrm{intLik}}$', color=colors[0], linestyle=':')
# plt.axvline(kk[np.log(likmg).mean(0).argmax()], label=r'$\theta_{\mathrm{E}}$', color=colors[1], linestyle=':')
# plt.legend()
# plt.xlabel(r'$\theta$')
# plt.ylabel(r'Likelihood')
# plt.subplot(1, 2, 2)
plt.plot(kk, -np.log(likmg.mean(0)), label=r'$-\log \mathcal{L}_{\mathrm{integrated}}(\theta;y)$')
plt.plot(kk, -np.log(likmg).mean(0), label=r'$\mu(\theta) = \mathrm{E}_U[J(\theta, U)]$')
plt.axvline(kk[likmg.mean(0).argmax()], label=r'$\hat{\theta}_{\mathrm{intMLE}}$', color=colors[0],
            linestyle=':')
plt.axvline(kk[np.log(likmg).mean(0).argmax()], label=r'$\hat{\theta}_{\mathrm{mean}}$', color=colors[1],
            linestyle=':')
plt.legend()
plt.xlabel(r'$\theta$')
plt.ylabel(r'$J$')
plt.tight_layout()
plt.savefig('./img/integrated_lik_average_costfunction.pgf')
plt.close()


# Pareto front ---------------------------------------------------------
plt.figure(figsize=col_full)
x = np.linspace(0, 1)
def fun_pareto(x):
    return 2 / (1 + x)**5
ax = plt.subplot(1, 1, 1)
plt.plot(x, fun_pareto(x), label=r'Pareto frontier')
plt.xlabel(r'$\mu(\theta)$')
plt.xticks([])
plt.yticks([])
plt.ylabel(r'$\sigma(\theta)$')
plt.plot(0.4, fun_pareto(0.4), 'o', color='green', label=r'$(\mu(\theta_0), \sigma(\theta_0))$')
ax.add_artist(patches.Rectangle((0.4, fun_pareto(0.4)), 1, 2, color='black', alpha=0.1))
plt.plot(0.5, 1, 'o', label = r'$(\mu(\theta_1), \sigma(\theta_1))$', color='red')
xx = 0.1
plt.plot(xx, fun_pareto(xx), 'o', color='green', label=r'$(\mu(\theta_2), \sigma(\theta_2))$')
ax.add_artist(patches.Rectangle((xx, fun_pareto(xx)), 1, 2, color='black', alpha=0.1))
plt.annotate(r'$\theta_0$', xy=(0.4, fun_pareto(0.4)), xytext=(0.42, fun_pareto(0.4)))
plt.annotate(r'$\theta_2$', xy=(xx, fun_pareto(xx)), xytext=(xx+0.02,fun_pareto(xx)))
plt.annotate(r'$\theta_1$', xy=(0.5, 1), xytext=(0.52, 1))
plt.title(u'Pareto frontier for \n the multiobjective optimization problem $(\mu, \sigma)$')
plt.legend()
plt.tight_layout()
plt.savefig('./img/pareto_frontier.pgf')
plt.close()




def asym_normal(x, position, scale, shape, displaymoments=False):
    if displaymoments:
        delt = shape / np.sqrt(1 + shape**2)
        mean = position + scale * np.sqrt(2 / np.pi) * delt
        var = scale**2 * (1 - 2 * delt**2 / np.pi)
        skew = (2 - np.pi / 2.0) * (delt * np.sqrt(2.0 / np.pi))**3 / (1 - 2 * delt**2 / np.pi)**(3.0 / 2.0)
        return mean, var, skew
    else:
        return (2.0 / scale) * scipy.stats.norm.pdf((x - position) / scale) * scipy.stats.norm.cdf(shape * (x - position) / scale)

x = np.linspace(-3, 4., 200)
xcdf = np.linspace(-3, 2., 200)
def compute_mean(asym):
    return np.sum(x * asym) * (x[1] - x[0])
    
alpha = -30
eta = 0
omega = 1
plt.figure(figsize=col_full)
plt.subplot(1, 2, 1)
asym = asym_normal(x, eta, omega, alpha)
plt.plot(x, asym, label=r'Skewness < 0')
mean, var, skew = asym_normal(0, eta, omega, alpha, True)
mean2, var2, skew2 = asym_normal(0, eta, omega, -alpha, True)

asym2 = asym_normal(x, eta - 2 * mean2, omega, -alpha)
plt.plot(x, asym2, label=r'Skewness > 0')

# plt.title(r'mean={}, var={}, skew={}'.format(mean, var, skew))
plt.plot(x, scipy.stats.norm.pdf(x, loc=mean, scale=np.sqrt(var)), label='Skewness = 0')
plt.title(u'Pdf of r.v. \n with different skewness')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(xcdf, np.cumsum(asym_normal(xcdf, eta, omega, alpha)) * (xcdf[1] - xcdf[0]), label='Skewness < 0')
plt.plot(xcdf, np.cumsum(asym_normal(xcdf, eta - 2 * mean2, omega, -alpha)) * (xcdf[1] - xcdf[0]), label='Skewness > 0')

# plt.title(r'mean={}, var={}, skew={}'.format(mean, var, skew))
plt.plot(xcdf, scipy.stats.norm.cdf(xcdf, loc=mean, scale=np.sqrt(var)), label='Skewness = 0')
# plt.xlabel(r'')
plt.legend()
plt.title(u'Cdf of r.v. \n with different skewness')
plt.tight_layout()
plt.savefig('./img/skewness_examples.pgf')
plt.close()



@np.vectorize
def branin_hoo(x, y):
    x1 = 3 * x * 5 - 5
    x2 = 3 * y * 5
    damp = 1.0
    quad = (x2 - (5.1 / (4 * np.pi**2)) * x1**2 + (5 / np.pi) * x1 - 6)**2
    cosi = (10 - (10 / np.pi * 8)) * np.cos(x1) - 44.81
    return (quad + cosi) / (51.95 * damp) + 2.0


@np.vectorize
def new_fun(x, y):
    return (1 + y * (x + 0.1)**2) * (1 + (x - y)**2)
    # return (x - np.sin(2 * np.pi * y) + 0.5)**2 - x

k = np.linspace(0, 1, 500)
u = np.linspace(0, 1, 500)
kmg, umg = np.meshgrid(k, u)
# bh = branin_hoo(kmg, umg)
# bh = -np.log(likmg)
bh = new_fun(kmg, umg)
regret = bh - bh.min(1)[:, np.newaxis]
plt.figure(figsize=col_full)
plt.subplot(1, 2, 1)
plt.contourf(k, u, bh)
# plt.scatter(k[bh.argmax(1)], u, marker='.', color=colors[0], label=r'$\max_u J$')
plt.scatter(k[bh.argmin(0)], u, marker='.', color='red', label=r'$\min_k J$')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend()
plt.title(r'Cost function')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$u$')
ax = plt.subplot(1, 2, 2)
# plt.contourf(k, u, regret)
# plt.scatter(k[regret.argmax(1)], u, marker='.', color='green', label=r'$\max_u r$')
ax.set_yticks([])
plt.plot(k, bh.max(0), label=r'$\max_{u} J(\theta,u)$', color=colors[0])
plt.plot(k, regret.max(0), label=r'$\max_{u} r(\theta,u)$', color=colors[1])
plt.plot(k, bh.min(0), label=r'$\min_{u} J(\theta, u)$', color=colors[2])
plt.legend()

plt.axvline(k[bh.max(0).argmin()], label=r'$\theta_{\mathrm{WC}}$', color=colors[0],
            linestyle=':')

plt.axvline(k[regret.max(0).argmin()], label=r'$\theta_{\mathrm{rWC}}$', color=colors[1],
            linestyle=':')
plt.axvline(k[bh.min(0).argmin()], label=r'$\theta_{\mathrm{global}}$', color=colors[2],
            linestyle=':')
ax.set_xlim([-.1, 1.1])
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'Criteria')
# ax2 = ax.twinx()
plt.title(u'Robust criteria')
plt.tight_layout()
# plt.show()
plt.savefig('./img/decision_under_uncertainty.pgf')
plt.close()
# ----------------------------------------------------------------------
## Example with branin: mean, worst case, sd
plt.figure(figsize=1 * col_full)
plt.subplot(1, 2, 1)
plt.contourf(k, u, bh)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$u$')
ax = plt.subplot(1, 2, 2)
axx, = ax.plot(k, bh.mean(0), label=r'$\mu(\theta)$', color=colors[0])
ax.set_yticks([])
plt.axvline(k[bh.mean(0).argmin()], label=r'$\theta_{\mathrm{mean}}$', color=colors[0],
            linestyle=':')
ax2 = ax.twinx()
ax2.set_yticks([])
ax2.plot(np.nan, np.nan, label=r'$\mu(\theta)$', color=colors[0])
ax2.plot(np.nan, np.nan, label=r'$\theta_{\mathrm{mean}}$', color=colors[0],
         linestyle=':')
axx, = ax2.plot(k, bh.std(0), label=r'$\sigma(\theta)$', color=colors[1])
plt.axvline(k[bh.std(0).argmin()], label=r'$\theta_{\mathrm{var}}$', color=colors[1],
            linestyle=':')

# ax3 = ax.twinx()
# ax3.set_yticks([])
# axx, = plt.plot(k, bh.max(0), label=r'$\max_{u} J(\theta,u)$', color=colors[2])
# plt.axvline(k[bh.max(0).argmin()], label=r'$\theta_{\mathrm{WC}}$', color=colors[2],
#             linestyle=':')
ax.set_xlabel(r'$\theta$')
ax.set_title(r'Mean and standard deviation')
plt.legend(loc=2)
plt.title(u'')
plt.tight_layout()
plt.savefig('./img/mean_std_wc.pgf')
plt.close()



#  ------------------------------------------------------------
plt.figure(figsize = col_full)
idx = [150, 300, 460]
threshold = 0.05

for i in range(3):
    plt.plot(k, bh[:, idx[i]], label=r'$J(\cdot, u_{})$'.format(i + 1), color=colors[i])
    a = np.where(bh[:, idx[i]] - bh[:, idx[i]].min() < threshold, True, False)
    print(sum(a))
    plt.plot(k[a], np.ones_like(k)[a] * (bh[:, idx[i]].min() + threshold), colors[i], linestyle=':')
    plt.plot(k[a], (0.75 - i / 10.0) * np.ones_like(k)[a], colors[i])
    plt.annotate(r'$\mathcal{{I}}_{{\beta}}(u_{})$'.format(i + 1),
                 (0.02 + k[a][-1], 0.75 - 0.1 * i))
plt.legend()
plt.xlabel(r'$\theta$')
plt.ylabel(r'$J$')
plt.tight_layout()
plt.savefig('./img/lik_interval_threshold.pgf')
plt.close()

# ----------------------------------------------------------------------
plt.figure(figsize = col_full)
idx = [150, 300, 460]
threshold = 0.05

plt.contourf(k, u, bh)
plt.legend()
plt.xlabel(r'$\theta$')
plt.ylabel(r'$J$')
plt.scatter(k[bh.argmin(0)], u, marker='.', color='red', label=r'$\min_k J$')
plt.contour(k, u, (bh - bh.min(0)).T < threshold, levels=[0.5])
plt.contour(k, u, (bh/bh.min(0)).T < 1.01, levels=[0.5])
plt.tight_layout()
# plt.show()
# plt.savefig('./img/lik_interval_threshold.pgf')
plt.close()





# Distribution of minimizers ------------------------------------------------------------

def minimizer_sample(func, Nsamples=300,
                     bounds_cal = [0, 1],
                     bounds_unc = [0, 1],
                     nrestart=4):
    mini = np.empty(Nsamples)
    uncc = np.empty(Nsamples)
    for i in range(Nsamples):
        unc = scipy.stats.uniform.rvs() * (bounds_unc[1] - bounds_unc[0]) + bounds_unc[0]
        uncc[i] = unc
        best = np.inf
        for j in range(nrestart):
            # x0 = scipy.stats.uniform.rvs() * (bounds_cal[1] - bounds_cal[0]) + bounds_cal[0]
            x0 = .8
            res_optim = scipy.optimize.minimize(func, args=(unc,), x0=x0,
                                                bounds=np.atleast_2d(bounds_unc))
            if res_optim.fun < best:
                mini[i] = res_optim.x
    return mini, uncc

import pandas as pd
import seaborn as sns
theta_star, u_sampled = minimizer_sample(lambda x, y: new_fun(y, x), Nsamples=2000, nrestart=1)
plt.figure(figsize=1 * col_full)

plt.subplot(1, 2, 1)
plt.contourf(k, u, bh)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$u$')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.scatter(theta_star, u_sampled, marker='.', color='red', label=r'$(\theta^*(u), u)$')
lleg = plt.legend()
for text in lleg.get_texts():
    text.set_color('white')
plt.title(r'Objective function $J$')
plt.subplot(1, 2, 2)
ax = sns.kdeplot(theta_star, label=r'$\hat{p}_{\theta^*}$')
plt.hist(theta_star, normed=True, alpha=.5, label=r'Histogram', color=colors[0])
ax.set_xlim([0, 1])
ax.set_xlabel(r'$\theta$')
ax.set_title(r'Estimation of $p_{\theta^*}$')
ax.set_ylabel(r'$\hat{p}_{\theta^*}$')
ax.set_ylim([0, 2.5])
ax.legend(loc='upper left')
# df = pd.DataFrame(np.asarray((theta_star, u_sampled)).T)
# df.columns = [r'$\theta^*$', r'$u$']
# sns.pairplot(data=df)
plt.tight_layout()
plt.savefig('./img/theta_star_samples.pgf')
# plt.show()
plt.close()


k = np.linspace(0, 1, 800)
u = np.linspace(0, 1, 500)
kmg, umg = np.meshgrid(k, u)
bh = branin_hoo(umg, kmg)

for j, cst in enumerate(['beta', 'alpha']):
    plt.figure(figsize=col_full)


    ax1 = plt.subplot(1, 2, 1)
    ax1.contourf(k, u, bh)
    # ax1.scatter(k[bh.argmin(1)], u, marker='.', s=5, color='red', label=r'$\min_k J$')
    ax2 = plt.subplot(1, 2, 2)
    if cst == 'beta':
        gen = enumerate(zip([.2, 0.5, 1], [1, 1, 1]))
        typ_reg = 'AR'
    else:
        gen = enumerate(zip([0, 0, 0], [1.1, 1.5, 2.0]))
        typ_reg = 'RR'
    for i, gr in gen:
        print(gr)
        Delta = bh - gr[1] * bh.min(1)[:, np.newaxis] - gr[0] < 0 
        ax1.contour(k, u, Delta, levels=[0.45, 0.55], cmap=ListedColormap(colors[i]))
        ax2.plot(k, Delta.mean(0), label=r'${{{c}}}_{{{index}}}$'.format(c='\\' + cst, index=i + 1))
        plt.vlines(k[Delta.mean(0).argmax()],
                   ymin=0, ymax=Delta.mean(0).max(), color=colors[i], ls=':',
                   label=r'$\hat{{\theta}}_{{\mathrm{{{typ}}}, {{{c}}}_{{{index}}}}}$'.format(c='\\' + cst, index=i + 1, typ=typ_reg))
    ax2.legend()
    ax2.set_xlabel(r'$\theta$')
    ax1.set_xlabel(r'$\theta$')
    ax1.set_ylabel(r'$u$')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax2.set_title(r'$\Gamma_{{{c}}}$ for increasing ${{{c}}}$'.format(c='\\' + cst))
    ax1.set_title(r'Regions of acceptability for increasing ${{{c}}}$'.format(c='\\' + cst))
    plt.tight_layout()
    plt.savefig('./img/gamma_{}_increasing.pgf'.format(cst))
    plt.close()



# @np.vectorize
# def new_fun(x, y):
#     return 1 + (100 * (1 + 2 * y)**2) + 1000 * (2 * (x - 0.2) - y)**2
#     # return (x - np.sin(2 * np.pi * y) + 0.5)**2 - x

k = np.linspace(0, 1, 800)
u = np.linspace(0, 1, 500)
kmg, umg = np.meshgrid(k, u)
# bh = branin_hoo(kmg, umg)
# bh = -np.log(likmg)
bh = new_fun(kmg, umg) * (0.1 + 10 * (umg)**2)
Delta = bh - bh.min(1)[:, np.newaxis] < 50
Delta = bh - bh.min(1)[:, np.newaxis] < 0.1

print(Delta.min(), Delta.max())
# regret = bh - bh.min(0)[:, np.newaxis]
plt.figure(figsize=col_full)
plt.subplot(1, 2, 1)
plt.contourf(k, u, bh)
# plt.scatter(k[bh.argmax(1)], u, marker='.', color=colors[0], label=r'$\max_u J$')
plt.scatter(k[bh.argmin(1)], u, marker='.', s=5, color='red', label=r'$\min_k J$')
plt.contour(k, u, Delta, levels=[0.45, 0.55], cmap=plt.get_cmap('gray'))
# plt.contour(k, u, bh - 1.2 * bh.min(1)[:, np.newaxis] < 0, levels=[0, 0.5, 1], cma=plt.get_cmap('RdPu'))
indU, indK = 200, 200
plt.plot(k[Delta[indU, :]], np.ones_like(k[Delta[indU, :]]) * u[indU],
         color=colors[0])
plt.plot(np.ones_like(u[Delta[:, indK]]) * k[indK], u[Delta[:, indK]],
         color=colors[1])
plt.annotate(s=r'$\mathcal{I}_{\beta}(u)$',
             xy=(k[indK] + 0.2, u[indU]), color='white')
plt.annotate(s=r'$R_{\beta}(\theta)$',
             xy=(k[indK] - 0.15, u[Delta[:, indK]][-1]), color='white')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel(r'$\theta$')
plt.ylabel(r'$u$')
plt.title(r'Additive regret')
plt.subplot(1, 2, 2)
Delta = bh - 1.03 * bh.min(1)[:, np.newaxis] < 0
plt.contourf(k, u, bh)
# plt.scatter(k[bh.argmax(1)], u, marker='.', color=colors[0], label=r'$\max_u J$')
plt.scatter(k[bh.argmin(1)], u, marker='.', s=5, color='red', label=r'$\min_k J$')
plt.contour(k, u, Delta, levels=[0.45, 0.55], cmap=plt.get_cmap('gray'))
# plt.contour(k, u, bh - 1.2 * bh.min(1)[:, np.newaxis] < 0, levels=[0, 0.5, 1], cma=plt.get_cmap('R(dPu'))
# indU, indK = 200, 360
plt.plot(k[Delta[indU, :]], np.ones_like(k[Delta[indU, :]]) * u[indU],
         color=colors[0])
plt.plot(np.ones_like(u[Delta[:, indK]]) * k[indK], u[Delta[:, indK]],
         color=colors[1])
plt.annotate(s=r'$\mathcal{I}_{\alpha}(u)$',
             xy=(k[indK] + 0.2, u[indU]), color='white')
plt.annotate(s=r'$R_{\alpha}(\theta)$',
             xy=(k[indK], u[Delta[:, indK]][-1]), color='white')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title(r'Relative regret')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$u$')
plt.tight_layout()
plt.savefig('./img/illustration_region_regret.pgf')
# plt.show()
plt.close()
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend()
plt.title(r'Cost function')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$u$')
ax = plt.subplot(1, 2, 2)
# plt.contourf(k, u, regret)
# plt.scatter(k[regret.argmax(1)], u, marker='.', color='green', label=r'$\max_u r$')
ax.set_yticks([])
plt.legend()

plt.axvline(k[bh.max(0).argmin()], label=r'$\theta_{\mathrm{WC}}$', color=colors[0],
            linestyle=':')
plt.axvline(k[regret.max(0).argmin()], label=r'$\theta_{\mathrm{rWC}}$', color=colors[1],
            linestyle=':')
plt.axvline(k[bh.min(0).argmin()], label=r'$\theta_{\mathrm{global}}$', color=colors[2],
            linestyle=':')
ax.set_xlim([-.1, 1.1])
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'Criteria')
# ax2 = ax.twinx()
plt.title(u'Robust criteria')
plt.tight_layout()
plt.close()


# EOF ----------------------------------------------------------------------

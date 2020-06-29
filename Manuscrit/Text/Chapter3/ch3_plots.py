#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import matplotlib as mpl
from matplotlib import patches
import matplotlib.pyplot as plt
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


## PDF and CDF of rv X
plt.figure(figsize=col_full)
@np.vectorize
def lik(k, u):
    return np.exp(-(k-(u-k)**2)**2/ 2) / np.sqrt(2 * np.pi)
kk, uu = np.linspace(-2, 2, 500), np.linspace(-2, 2, 400)
kmg, umg = np.meshgrid(kk, uu)
likmg = lik(kmg, umg)
plt.subplot(2, 1, 1)
plt.contourf(lik(kmg, umg))
plt.title(r'Likelihood: $p_{Y \mid \theta, U}$')
plt.subplot(2, 1, 2)
plt.plot(likmg.mean(0), label=r'Integrated Likelihood')
plt.plot(likmg.max(1), label=r'Profile Likelihood')
plt.legend()
plt.show()
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
plt.ylabel(r'$\sigma^2(\theta)$')
plt.plot(0.4, fun_pareto(0.4), 'o', color='green', label=r'$(\mu(\theta_0), \sigma^2(\theta_0))$')
ax.add_artist(patches.Rectangle((0.4, fun_pareto(0.4)), 1, 2, color='black', alpha=0.1))
plt.plot(0.5, 1, 'o', label = r'$(\mu(\theta_1), \sigma^2(\theta_1))$', color='red')
xx = 0.1
plt.plot(xx, fun_pareto(xx), 'o', color='green', label=r'$(\mu(\theta_2), \sigma^2(\theta_2))$')
ax.add_artist(patches.Rectangle((xx, fun_pareto(xx)), 1, 2, color='black', alpha=0.1))
plt.annotate(r'$\theta_0$', xy=(0.4, fun_pareto(0.4)), xytext=(0.42, fun_pareto(0.4)))
plt.annotate(r'$\theta_2$', xy=(xx, fun_pareto(xx)), xytext=(xx+0.02,fun_pareto(xx)))
plt.annotate(r'$\theta_1$', xy=(0.5, 1), xytext=(0.52, 1))
plt.title(u'Pareto frontier for \n the multiobjective problem $[\mu, \sigma^2]$')
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


## Example with branin
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

@np.vectorize
def branin_hoo(y, x):
    x1 = 3 * x * 5 - 5
    x2 = 3 * y * 5
    damp = 1.0
    quad = (x2 - (5.1 / (4 * np.pi**2)) * x1**2 + (5 / np.pi) * x1 - 6)**2
    cosi = (10 - (10 / np.pi * 8)) * np.cos(x1) - 44.81
    return (quad + cosi) / (51.95 * damp) + 2.0

k = np.linspace(0, 1, 200)
u = np.linspace(0, 1, 200)
kmg, umg = np.meshgrid(k, u)
bh = branin_hoo(kmg, umg)
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
axx, = ax2.plot(k, bh.std(0), label=r'$\sigma(\theta)$', color=colors[1])
plt.axvline(k[bh.std(0).argmin()], label=r'$\theta_{\mathrm{var}}$', color=colors[1],
            linestyle=':')

ax3 = ax.twinx()
ax3.set_yticks([])
axx, = plt.plot(k, bh.max(0), label=r'$\max_{u} J(\theta,u)$', color=colors[2])
plt.axvline(k[bh.max(0).argmin()], label=r'$\theta_{\mathrm{WC}}$', color=colors[2],
            linestyle=':')
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'Robust quantity')
plt.legend()
plt.title(u'')
plt.tight_layout()
plt.savefig('./img/mean_std_wc.pgf')
plt.close()
# EOF ----------------------------------------------------------------------


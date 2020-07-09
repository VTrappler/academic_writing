#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import matplotlib as mpl
# from matplotlib import cm
# import matplotlib.pyplot as plt
# import numpy as np
# import scipy.stats
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
ax.plot([1], [0.5], markersize = 5, marker='o', color='r')
ax.annotate('', xy=(1.0, 0.5), xytext=(1.0, 0.0),
            arrowprops={'arrowstyle': '-|>'}, va='center')

ax.legend()
plt.savefig('./img/cdf_pdf_example.pgf')
plt.close()
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
ax.set_title(r'Pdf of a 2D-Gaussian r.v.')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
plt.savefig('./img/gaussian_distribution_examples.pgf')
plt.close()
# ----------------------------------------------------------------------
# Pdf of chi2
plt.figure(figsize=col_3quarter)
x = np.linspace(0, 10, 200)
plt.ylim([0, .75])
plt.xlim([0, 8.5])
plt.plot(x, scipy.stats.chi2.pdf(x, 1), label=r'$\nu=1$')
plt.plot(x, scipy.stats.chi2.pdf(x, 2), label=r'$\nu=2$')
plt.plot(x, scipy.stats.chi2.pdf(x, 4), label=r'$\nu=4$')
plt.plot(x, scipy.stats.chi2.pdf(x, 6), label=r'$\nu=6$')
plt.xlabel(r'$x$')
plt.ylabel(r'$p_X$')
plt.legend()
plt.title(r'Pdf of $X \sim $\chi^2_{\nu}$, depending on $\nu$')
# plt.show()
plt.tight_layout()
plt.savefig('./img/chi2_distribution_examples.pgf')
plt.close()


plt.figure(figsize=col_3quarter)
x = np.linspace(4, 7, 1000)
lik = scipy.stats.norm.pdf(x, 5, 0.2) + scipy.stats.norm.pdf(x, 5.15, .1)
maxLik = lik.max()
lik = lik / maxLik
plt.plot(x, lik, label=r'$R(\theta)$')
xrel = lik > 0.15
plt.plot(x[xrel], 0.15 * np.ones_like(x[xrel]), 'r', label=r'$\mathcal{I}_{\mathrm{lik}}(p)$')
plt.axvline(x[lik.argmax()], linestyle=':', label=r'$\hat{\theta}_{\mathrm{MLE}}$')
plt.legend()
# plt.xticks([])
plt.xlabel(r'$\theta$')
plt.ylabel(r'$R(\theta)$')
plt.title(r'Relative Likelihood and likelihood interval')
plt.tight_layout()
# plt.show()
plt.savefig('./img/relative_likelihood.pgf')
plt.close()


#  MLE on regression----------------------------------------------------
np.random.seed(3367)
eps = scipy.stats.norm.rvs(size=11)
x = np.linspace(0, 10, 11)
a, b = .9 , 0.3
c, d = 0.8, 0

y = a * x + b + eps

y = a * x + b * np.log(x+1) + eps
plt.figure(figsize=col_full)
plt.subplot(1, 2, 1)
plt.plot(x, y, 'o', label=r'samples')
plt.plot(x, a * x + b, label=r'truth')
plt.plot(x, 1 * x + 0, label=r'$y = x$')

# y = a*x + b + N(0, 1)

@np.vectorize
def lik(a, b):
    return  np.exp(-0.5 * np.sum((a * x + b - y)**2, 0)) / np.sqrt(2 * np.pi)**(len(y))

optim = scipy.optimize.minimize(lambda x: -np.log(lik(x[0], x[1])), x0 = [0, 0])
MLE = optim.x
plt.plot(x, MLE[0] * x + MLE[1], label=r'$MLE$')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Linear regression and model selection')
plt.legend()


aa, bb = np.meshgrid(np.linspace(0.5, 1, 400), np.linspace(-2, 2, 400))
plt.subplot(1, 2, 2)
likgrid = lik(aa, bb)
plt.contourf(np.linspace(0.5, 1, 400), np.linspace(-2, 2, 400), lik(aa, bb))
plt.plot(a, b, 'o', color='r')
plt.plot(c, d, 'o', color='g')
plt.plot(MLE[0], MLE[1], 'o', color='y')
plt.xlabel(r'$\theta_1$')
plt.ylabel(r'$\theta_2$')
plt.title(r'Likelihood')
plt.tight_layout()
plt.close()

plt.subplot(2, 1, 1)
plt.contourf(np.linspace(0.5, 1, 400), np.linspace(-2, 2, 400), likgrid)
plt.subplot(2, 1, 2)
plt.plot(np.linspace(0.5, 1, 400), likgrid.mean(0))
plt.plot(np.linspace(0.5, 1, 400), likgrid.max(0))
plt.close()





colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

@np.vectorize
def branin_hoo(x, y):
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
i1 = 40
i2 = 50
i3 = 180
u1 = u[i1]
u2 = u[i2]
u3 = u[i3]
icolor_st = 2
plt.figure(figsize=1 * col_full)
plt.subplot(1, 2, 1)
plt.contourf(k, u, bh)
plt.axhline(u1, color=colors[icolor_st])
plt.axhline(u2, color=colors[icolor_st + 1])
plt.axhline(u3, color=colors[icolor_st + 2])

plt.xlabel(r'$\theta$')
plt.ylabel(r'$u$')
plt.yticks([u1, u2, u3], [r'$u_0$', r'$u_1$', r'$u_2$'])
plt.xticks([])
plt.title(r'Contour of $J(\theta,u)$')
ax = plt.subplot(1, 2, 2)
axx, = ax.plot(k, bh[i1], label=r'$J(\theta, u_0)$', color=colors[icolor_st])
plt.axvline(k[bh[i1].argmin()], # label=r'$\hat{\theta}(u_0)$,'
            color=colors[icolor_st],
            linestyle=':')
axx, = ax.plot(k, bh[i2], label=r'$J(\theta, u_1)$', color=colors[icolor_st + 1])
plt.axvline(k[bh[i2].argmin()], # label=r'$\hat{\theta}(u_1)$',
            color=colors[icolor_st + 1],
            linestyle=':')

axx, = ax.plot(k, bh[i3], label=r'$J(\theta, u_2)$', color=colors[icolor_st + 2])
plt.axvline(k[bh[i3].argmin()], # label=r'$\hat{\theta}(u_2)$',
            color=colors[icolor_st + 2],
            linestyle=':')


plt.xticks([k[bh[i1].argmin()], k[bh[i2].argmin()], k[bh[i3].argmin()]], [r'$\hat{\theta}(u_0)$', r'$\hat{\theta}(u_1)$', r'$\hat{\theta}(u_2)$'])

ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$J(\theta, u)$')
plt.legend()
plt.title(r'$J(\theta, u)$ for fixed $u$')
plt.tight_layout()
plt.savefig('./img/minimizer_misspecification.pgf')
plt.close()
# EOF ----------------------------------------------------------------------

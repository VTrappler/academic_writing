#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.colors as colorz
import cmocean
import numpy as np
from netCDF4 import Dataset
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker

exec(open('/home/victor/acadwriting/Manuscrit/plots_settings.py').read())



plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=10.0)
plt.rc('legend', fontsize=10.0)
plt.rc('font', weight='normal')


def add_all_decorations(ax):
    lonmin = -9.
    lonmax = 1.
    latmin = 43.
    latmax = 51.

    ax.set_extent([lonmin, lonmax, latmin, latmax], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale('10m'), zorder=100, edgecolor='k')
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), zorder=101)
    ax.add_feature(cfeature.RIVERS.with_scale('10m'), zorder=101)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':', zorder=102)
    ax.coastlines(resolution='50m')
    ax.set_ylim([43, 51])
    gl = ax.gridlines(alpha=0.1, draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator([-10, -8, -6, -4, -2, 0, 2])
    gl.ylocator = mticker.FixedLocator([42, 44, 46, 48, 50, 52])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}
    ax.set_aspect(1)


def main():

    fig = plt.figure(figsize=(col_full[0], col_full[1]*0.6))
    ax1 = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax2 = plt.subplot(1, 2, 2, projection=ccrs.PlateCarree())
    croco_grd = Dataset('/home/victor/sandbox/CROCO_FILES_test/croco_grd_gaussian.nc')
    # logdepth = -np.log(croco_grd['h'][:])
    depth = (croco_grd['h'][:])
    
    lon = croco_grd['lon_rho'][:]
    lat = croco_grd['lat_rho'][:]
    croco_grd.close()
    lonmin = -9.
    lonmax = 1.
    latmin = 43.
    latmax = 51.



    im1 = ax1.contourf(lon, lat, depth, 10, transform=ccrs.PlateCarree(), cmap=cmocean.cm.thermal_r)
    im2 = ax2.contourf(lon, lat, depth, 50, levels=[1e1, 3e1, 5e1, 1e2, 5e2, 1e3, 1e4], transform=ccrs.PlateCarree(),
                       norm=colorz.LogNorm(), cmap=cmocean.cm.thermal_r)
    cb1 = plt.colorbar(im1, ax=ax1)
    # cb1.ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    # cb1.ax.ticklabel_format(style='sci', axis='y', useOffset=True, scilimits=(0, 0))
    
    plt.colorbar(im2, ax=ax2)
    ax1.set_title('Ocean floor elevation (in $\mathrm{m}$)')
    ax2.set_title('Ocean floor elevation (log-scale)')
    #ax.add_feature(cfeature.OCEAN.with_scale('50m'))
    for ax in (ax1, ax2):
        add_all_decorations(ax)

        # ax.set_adjustable('datalim')
    plt.tight_layout()
    plt.savefig('./img/depth_maps.pdf')
    # plt.show()
    plt.close()

    
    fig = plt.figure(figsize=(col_full[0], col_full[1]))
    ax1 = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    im1 = ax1.contourf(lon, lat, depth, 50, levels=[1e1, 3e1, 5e1, 1e2, 5e2, 1e3, 5e3],
                       transform=ccrs.PlateCarree(),
                       norm=colorz.LogNorm(), cmap=cmocean.cm.deep)
    formatter = mticker.ScalarFormatter(# 10, labelOnlyBase=False
    )
    cbar = fig.colorbar(im1, ax=ax1, ticks=[10, 30, 50, 100, 500, 1000, 5000], format=formatter)
    # cb1.ax.ticklabel_format(style='sci', axis='y', useOffset=True, scilimits=(0, 0))
    ax1.text(-6, 46, 'Bay of Biscay', color='white')
    ax1.text(-8, 50, 'Celtic Sea', color='white')
    ax1.text(-3, 50, 'English Channel', color='white')

    ax1.set_title('Ocean floor elevation (in $\mathrm{m}$)')
    add_all_decorations(ax1)
    plt.tight_layout()
    plt.savefig('./img/depth_maps_log.png', dpi=300)
    plt.show()
    plt.close()

    
    # fig = plt.figure(figsize=(col_full[0], col_full[1]))
    # ax1 = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    # im1 = ax1.contourf(lon, lat, np.asarray(croco_grd['z0b']).squeeze(), 10, transform=ccrs.PlateCarree())
    # add_all_decorations(ax1)
    # plt.colorbar(im1, ax=ax1, format='%.0e')
    # plt.title(r'\textsf{True value of }$\mathsf{\theta}$ \textsf{(in $\mathrm{m}$)}')
    # plt.savefig('./img/gaussian_english_channel.png', dpi=300)
    # plt.close()

    # # fname = '/home/victor/croco_dahu/croco-tap/croco/RunC/z0b.00-010'
    # # with open(fname, 'r') as ff:
    # #     lines = ff.readlines()[1:]

    # # z0b = np.empty((len(lines), 3))
    # # for i, ll in enumerate(lines):
    # #     z0b[i, :] = np.fromiter(map(float, ll.split()), dtype=float)
    # import csv
    # import pandas as pd
    # fig = plt.figure(figsize=(col_full[0], col_full[1]))
    # # SA_results = np.empty((6, 6))
    # # with open('/home/victor/acadwriting/Manuscrit/Text/Chapter5/SA_croco.csv') as fi:
    # #     SAfile = csv.reader(fi)
    # #     SAfile.next()
    # #     for i, row in enumerate(SAfile):
    # #         print(row)

    # df = pd.read_csv('/home/victor/acadwriting/Manuscrit/Text/Chapter5/SA_croco.csv', header=0)
    # colnames = ['num', 'Sobol indice', r'$D_1$', r'$D_2$', r'$D_3$', r'$D_4$', r'$u_1$', r'$u_2$']
    # df = df.rename(dict(zip(df.columns.values, colnames)), axis='columns')
    # df = df.drop(columns='num')
    # df = df.melt(id_vars=['Sobol indice'])

    # import seaborn as sns

    # df['Sobol indice'][df['Sobol indice'] == 'S2'] = r'$S_2$'
    # df['Sobol indice'][df['Sobol indice'] == 'S'] = r'$S_1$'
    # df['Sobol indice'][df['Sobol indice'] == 'T'] = r'$S_T$'
    # sns.barplot(x = 'variable', y='value', hue='Sobol indice', data=df)
    # plt.title(r'Sobol indices for CROCO')
    # plt.xlabel(r'Sobol indices')

    # plt.savefig('./img/SA_croco.pgf')
    # plt.close()

    # fig = plt.figure(figsize=(col_full[0], col_full[1]))
    # SA_results = np.empty((6, 6))
    # with open('/home/victor/acadwriting/Manuscrit/Text/Chapter5/SA_croco.csv') as fi:
    #     SAfile = csv.reader(fi)
    #     SAfile.next()
    #     for i, row in enumerate(SAfile):
    #         print(row)

    # df = pd.read_csv('/home/victor/acadwriting/Manuscrit/Text/Chapter5/tides_T.csv', header=0)

    # print(df)
    # sns.barplot(x = 'Unnamed: 0', y='original', data=df)
    # plt.title(r'Sobol indices for CROCO')
    # plt.xlabel(r'Sobol indices')
    # plt.show()


def main2():
    von_karman = 0.4


    def Cb(H, z0b):
        return (von_karman / np.log(H / z0b))**2
    H = 100
    z0b = np.logspace(-6, -1)
    plt.figure(figsize=col_full)
    plt.plot(z0b, Cb(10, z0b), label=r'$H=\SI{10}{\meter}$')
    plt.plot(z0b, Cb(100, z0b), label=r'$H=\SI{100}{\meter}$')
    plt.plot(z0b, Cb(500, z0b), label=r'$H=\SI{500}{\meter}$')
    plt.plot(z0b, Cb(2000, z0b), label=r'$H=\SI{2000}{\meter}$')
    plt.plot(z0b, Cb(5000, z0b), label=r'$H=\SI{5000}{\meter}$')
    bounds = [20e-3, 2e-3, 0.5e-3, 0.05e-3, 0.01e-3]
    for bd in bounds:
        plt.axvline(x=bd, color='k', alpha=0.3)
    plt.text(.8e-6, 0.007, 'Silts and Clay')
    plt.text(1.5e-5, 0.0068, 'Fine\nsand')
    plt.text(1e-4, 0.007, 'Sand')
    plt.text(6e-4, 0.007, 'Gravel')
    plt.text(4e-3, 0.007, 'Pebble')
    plt.text(4e-2, 0.007, 'Rocks')

    plt.xscale('log')
    plt.ylabel(r'$C_d$')
    plt.xlabel(r'$z_{0,b}$ (in m)')
    plt.title('Bottom friction as a function of the roughness\nand the water column')
    plt.legend(loc='center left', fontsize=8)
    plt.tight_layout()
    plt.savefig('/home/victor/acadwriting/Manuscrit/Text/Chapter5/img/cd_zob.pgf')
    plt.show()

    
if __name__ == '__main__':
    main()                      #
    # main2()

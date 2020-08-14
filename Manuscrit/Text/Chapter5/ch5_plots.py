#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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

    lonmin = -9.
    lonmax = 1.
    latmin = 43.
    latmax = 51.



    im1 = ax1.contourf(lon, lat, depth, 10, transform=ccrs.PlateCarree())
    im2 = ax2.contourf(lon, lat, depth, 50, transform=ccrs.PlateCarree(),
                       norm=colors.LogNorm())
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
    plt.show()
    plt.close()

    
    fig = plt.figure(figsize=(col_full[0], col_full[1]))
    ax1 = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    im1 = ax1.contourf(lon, lat, np.asarray(croco_grd['z0b']).squeeze(), 10, transform=ccrs.PlateCarree())
    add_all_decorations(ax1)
    plt.colorbar(im1, ax=ax1, format='%.0e')
    plt.title(r'True value of $z_0$ (in $\mathrm{m}$)')
    plt.savefig('./img/gaussian_english_channel.pdf')
    plt.close()

    
    # fname = '/home/victor/croco_dahu/croco-tap/croco/RunC/z0b.00-010'
    # with open(fname, 'r') as ff:
    #     lines = ff.readlines()[1:]

    # z0b = np.empty((len(lines), 3))
    # for i, ll in enumerate(lines):
    #     z0b[i, :] = np.fromiter(map(float, ll.split()), dtype=float)
    
if __name__ == '__main__':
    main()

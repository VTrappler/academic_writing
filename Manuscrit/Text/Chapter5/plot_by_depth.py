#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import netCDF4 as netcdf
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


nctarget = netcdf.Dataset('/home/victor/sandbox/CROCO_FILES_test/croco_grd_template.nc',
                          'r', format='NETCDF4')
bathy = nctarget['h'][:]
nctarget.close()

depth_bins = [10., 20, 30, 50, 100, 200, 5000]
# depth_bins = [0., 30, 100, 5000]
depth_bins = [0, 60, 100, 200, 5000] # For SA
depth_bins = [0, 30, 60, 100, 5000] # For SA

def map_grid_by_depth(z0bvalues, depth_idx):
    z0barray = np.zeros_like(bathy)
    # print('depth_idx={}'.format(len((depth_idx))))
    # print('z0bvalues={}'.format(len(z0bvalues)))
    if len(depth_idx) != len(z0bvalues):
        # print('TODO: add error message')
        pass
    else:
        # print('ifentered')
        for i, z in enumerate(z0bvalues):
        #    print(i)
        #    print(depth_idx[i])
            z0barray[depth_idx[i]] = z
    return z0barray


def make_grid_by_depth(z0bvalues, depth_bins, plot=False, grd_file_modif=False):
    nctarget = netcdf.Dataset('/home/victor/sandbox/CROCO_FILES_test/croco_grd_template.nc',
                          'r', format='NETCDF4')
    bathy = nctarget['h'][:]
    nctarget.close()
    idx = np.full((len(depth_bins) - 1, bathy.shape[0], bathy.shape[1]), False, dtype=bool)
    print(len(depth_bins))
    for j in range(len(depth_bins) - 1):
        idx[j] = np.logical_and((bathy >= depth_bins[j]), (bathy < depth_bins[j + 1]))
    z0barray = map_grid_by_depth(z0bvalues, depth_idx=idx)
    if plot:
        for j in range(len(depth_bins) - 1):
            plt.subplot(3, 2, j + 1)
            plt.contourf(idx[j])
        plt.show()
        plt.contourf(z0barray)
        for i in range(len(depth_bins) - 1):
            plt.contour(idx[i])
        plt.show()
    if grd_file_modif:
        make_grid.make_grid(z0barray)
    else:
        return z0barray


z0bvals = np.array([0, 1, 2, 3])
print(z0bvals)

z0barray = make_grid_by_depth(z0bvals, depth_bins, plot=True, grd_file_modif=False)

croco_grd = netcdf.Dataset('/home/victor/sandbox/CROCO_FILES_test/croco_grd_gaussian.nc')
    # logdepth = -np.log(croco_grd['h'][:])
depth = (croco_grd['h'][:])
lon = croco_grd['lon_rho'][:]
lat = croco_grd['lat_rho'][:]



lonmin = -9.
lonmax = 1.
latmin = 43.
latmax = 51.


n_int = len(depth_bins) - 1
cmap = plt.get_cmap('Set2', n_int)
# norm = mpl.colors.BoundaryNorm(np.arange(-0.5,3), cmap.N)

fig = plt.figure(figsize=(col_full[0], col_full[1]))
ax1 = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
plt.subplots_adjust(left=0.1, right=.75)
cbar_ax = fig.add_axes([0, 0, 0.1, 0.1])


im1 = ax1.contourf(lon, lat, z0barray, 10, transform=ccrs.PlateCarree(), cmap=cmap)
resize_colorbar = get_resize_event_function(ax1, cbar_ax)
fig.canvas.mpl_connect('resize_event', resize_colorbar)

cbar = plt.colorbar(im1, ax=ax1, cax=cbar_ax)
tick_locs = (np.arange(n_int) + 0.5) * (n_int - 1) / n_int
cbar.set_ticks(tick_locs)
labels = [r'$\mathsf{{D}}_{2}: {0}\leq \mathsf{{h}} <{1}$'.format(depth_bins[i], depth_bins[i + 1], i + 1) for i in range(n_int)]
cbar.ax.set_yticklabels(labels)
add_all_decorations(ax1)
ax1.set_title(r'\textsf{Repartition of the depth of the ocean bed}')
plt.tight_layout()
resize_colorbar(None)

plt.savefig('./img/depth_repartition.pdf')
plt.show()

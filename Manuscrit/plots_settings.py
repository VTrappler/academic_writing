#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function, with_statement
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import sys
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
# -> Manuscrit 415.41025
# -> Notes 418.25368


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
if sys.version_info >= (3, 0):
    plt.rc('text.latex', preamble=r"\usepackage{amsmath} \usepackage{amssymb}")
else:
    plt.rc('text.latex', preamble=b"\usepackage{amsmath} \usepackage{amssymb}")
    # mpl.use('pgf')
plt.style.use('seaborn')
plt.rc('font', **{'family': 'serif',
                  'serif': ['Computer Modern Roman']})
plt.rcParams.update(params)

plt.rc('text', usetex=True)
col_half = get_figsize()
col_full = get_figsize(wf=1.0)
col_3quarter = get_figsize(wf=.75)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


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


def add_colorbar_subplot(mappable, format):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # if im is None:
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax, format=format)
    plt.sca(last_axes)
    return cbar


def get_resize_event_function(ax, cbar_ax):
    """
    Returns a function to automatically resize the colorbar
    for cartopy plots
    
    Parameters
    ----------
    ax : axis
    cbar_ax : colorbar axis
    
    Example
    -------
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt
    
        fig, ax = plt.subplots(figsize=(10,5), subplot_kw={'projection': ccrs.PlateCarree()})
        cbar_ax = fig.add_axes([0, 0, 0.1, 0.1])
        
        [... your code generating a scalar mappable ...]
    
        resize_colorbar = get_resize_event_function(ax, cbar_ax)
        fig.canvas.mpl_connect('resize_event', resize_colorbar)
    
    Credits
    -------
    Solution by pelson at http://stackoverflow.com/a/30077745/512111
    """
    def resize_colorbar(event):
        plt.draw()
        posn = ax.get_position()
        cbar_ax.set_position([posn.x0 + posn.width + 0.01, posn.y0,
                              0.02, posn.height])
    return resize_colorbar



def progressbar(it, prefix="", size=60, file=sys.stdout, hide=False):
    if hide:
        for i, item in enumerate(it):
            yield item
    else:
        count = len(it)
        def show(j):
            x = int(size * j / count)
            file.write("%s[%s%s] %i/%i\r" % (prefix, "#" * x, "." * (size - x), j, count))
            file.flush()
        show(0)
        for i, item in enumerate(it):
            yield item
            show(i + 1)
        file.write("\n")
        file.flush()

import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
from matplotlib.artist import setp
from matplotlib.collections import PatchCollection
import os
from shapely.geometry import Point, Polygon, LineString
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter

# Folders containing the field data averaged over time and sample for different scenarios.
mylist = ['vd1_A2000', 'vd2_A1750', 'vd3_A1500', 'vd4_A1250', 'taset0', 'taset80', 'taset150', 'taset220', 'taset350', 'taset500']

# Figure titles
titles = [r'$v^0$=1', r'$v^0$=2', r'$v^0$=3', r'$v^0$=4', r'$T_{ASET}$' + '\n' + '=0', r'$T_{ASET}$' + '\n' + '=80', r'$T_{ASET}$' + '\n' + '=150', r'$T_{ASET}$' + '\n' + '=220', r'$T_{ASET}$' + '\n' + '=350', r'$T_{ASET}$' + '\n' + '=500']

# Interval titles
interval_titles = [r'$|N|$' + '\n' + '$\in$[190,135)', r'$|N|$' + '\n' + '$\in$[135,90)', r'$|N|$' + '\n' + '$\in$[90,55)', r'$|N|$' + '\n' + '$\in$[55,10)']

# Number of scenarios
n_scenarios = 10

# Number of intervals
n_intervals = 4

# Figure for temporal evolution of speed
fig0 = plt.figure(0)
grid0 = ImageGrid(fig0, 111,
                  nrows_ncols=(n_scenarios,n_intervals),
                  axes_pad=0.1,
                  share_all=True,
                  cbar_location="right",
                  cbar_mode="single",
                  cbar_size="3%",
                  cbar_pad="2%",
                  )

# Figure for temporal evolution of density
fig1 = plt.figure(1)
grid1 = ImageGrid(fig1, 111,
                  nrows_ncols=(n_scenarios,n_intervals),
                  axes_pad=0.1,
                  share_all=True,
                  cbar_location="right",
                  cbar_mode="single",
                  cbar_size="3%",
                  cbar_pad="2%",
                  )

# Figure for temporal evolution of radial speed
fig2 = plt.figure(2)
grid2 = ImageGrid(fig2, 111,
                  nrows_ncols=(n_scenarios,n_intervals),
                  axes_pad=0.1,
                  share_all=True,
                  cbar_location="right",
                  cbar_mode="single",
                  cbar_size="3%",
                  cbar_pad="2%",
                  )

# Figure for temporal evolution of pressure
fig3 = plt.figure(3)
grid3 = ImageGrid(fig3, 111,
                  nrows_ncols=(n_scenarios,n_intervals),
                  axes_pad=0.1,
                  share_all=True,
                  cbar_location="right",
                  cbar_mode="single",
                  cbar_size="3%",
                  cbar_pad="2%",
                  )

# Figure for temporal evolution of radial pressure
fig4 = plt.figure(4)
grid4 = ImageGrid(fig4, 111,
                  nrows_ncols=(n_scenarios,n_intervals),
                  axes_pad=0.1,
                  share_all=True,
                  cbar_location="right",
                  cbar_mode="single",
                  cbar_size="3%",
                  cbar_pad="2%",
                  )

# Colormaps should be chosen so that they present the information in the figure in a most accurate way.
cmap0 = plt.cm.viridis
cmap1 = plt.cm.viridis
cmap2 = plt.cm.viridis
cmap3 = plt.cm.viridis
cmap4 = plt.cm.inferno

# Loop over all scenarios
for i in range(0, n_scenarios):

    # Loop over all intervals
    for j in range(1, n_intervals+1):

        # Open average speed file for particular scenario and interval
        with h5py.File("{}{}{}{}{}{}".format('average_fields/', mylist[i], '/', 'average_speed_interval', j, '.hdf5'),
                       'r') as hf1:
            avg_speed = hf1['average_fields'][mylist[i]]["{}{}{}".format('average_speed_interval', j, '.npy.gz')][()]

        # Open average density file for particular scenario and interval
        with h5py.File("{}{}{}{}{}{}".format('average_fields/', mylist[i], '/', 'average_density_interval', j, '.hdf5'),
                       'r') as hf2:
            avg_density = hf2['average_fields'][mylist[i]]["{}{}{}".format('average_density_interval', j, '.npy.gz')][()]

        # Open average radial speed file for particular scenario and interval
        with h5py.File("{}{}{}{}{}{}".format('average_fields/', mylist[i], '/', 'average_radial_speed_interval', j, '.hdf5'),
                       'r') as hf3:
            avg_radial_speed = hf3['average_fields'][mylist[i]]["{}{}{}".format('average_radial_speed_interval', j, '.npy.gz')][()]

        # Open average pressure file for particular scenario and interval
        with h5py.File("{}{}{}{}{}{}".format('average_fields/', mylist[i], '/', 'average_pressure_interval', j, '.hdf5'),
                       'r') as hf4:
            avg_pressure = hf4['average_fields'][mylist[i]]["{}{}{}".format('average_pressure_interval', j, '.npy.gz')][()]

        # Open average radial pressure file for particular scenario and interval
        with h5py.File("{}{}{}{}{}{}".format('average_fields/', mylist[i], '/', 'average_radial_pressure_interval', j, '.hdf5'),
                       'r') as hf5:
            avg_radial_pressure = hf5['average_fields'][mylist[i]]["{}{}{}".format('average_radial_pressure_interval', j, '.npy.gz')][()]

        # Grid indexing
        grid_indx = i*n_intervals + j - 1

        # Plot speed for particular scenario and interval
        im0 = grid0[grid_indx].imshow(avg_speed.T, interpolation='bilinear', cmap=cmap0, vmin=0, vmax=1.4)
        if j==1:
            grid0[grid_indx].set_ylabel(titles[i], fontsize=11, position=(1,0.5), verticalalignment='bottom')
        if grid_indx in {0, 1, 2, 3}:
            grid0[grid_indx].set_title(interval_titles[j-1], fontsize=11)
        grid0[grid_indx].set_yticks([200, 175, 150, 125, 100, 75, 50, 25, 0])
        grid0[grid_indx].set_xticks([200, 175, 150, 125, 75, 50, 25, 0])
        grid0[grid_indx].set_yticklabels([])
        grid0[grid_indx].set_xticklabels([])
        grid0[grid_indx].get_yaxis().set_tick_params(direction='out', width=1, length=2, top='off')
        grid0[grid_indx].get_xaxis().set_tick_params(direction='out', width=1, length=2, top='off')
        grid0[grid_indx].set_ylim([200, 125])
        grid0[grid_indx].set_xlim([25, 175])

        # Plot density for particular scenario and interval
        im1 = grid1[grid_indx].imshow(avg_density.T, interpolation='bilinear', cmap=cmap1, vmin=0, vmax=3)
        if j==1:
            grid1[grid_indx].set_ylabel(titles[i], fontsize=11, position=(1,0.5), verticalalignment='bottom')
        if grid_indx in {0, 1, 2, 3}:
            grid1[grid_indx].set_title(interval_titles[j-1], fontsize=11)
        grid1[grid_indx].set_yticks([200, 175, 150, 125, 100, 75, 50, 25, 0])
        grid1[grid_indx].set_xticks([200, 175, 150, 125, 75, 50, 25, 0])
        grid1[grid_indx].set_yticklabels([])
        grid1[grid_indx].set_xticklabels([])
        grid1[grid_indx].get_yaxis().set_tick_params(direction='out', width=1, length=2, top='off')
        grid1[grid_indx].get_xaxis().set_tick_params(direction='out', width=1, length=2, top='off')
        grid1[grid_indx].set_ylim([200, 125])
        grid1[grid_indx].set_xlim([25, 175])

        # Plot radial speed for particular scenario and interval
        im2 = grid2[grid_indx].imshow(avg_radial_speed.T, interpolation='bilinear', cmap=cmap2, vmin=0, vmax=1)
        if j==1:
            grid2[grid_indx].set_ylabel(titles[i], fontsize=11, position=(1,0.5), verticalalignment='bottom')
        if grid_indx in {0, 1, 2, 3}:
            grid2[grid_indx].set_title(interval_titles[j-1], fontsize=11)
        grid2[grid_indx].set_yticks([200, 175, 150, 125, 100, 75, 50, 25, 0])
        grid2[grid_indx].set_xticks([200, 175, 150, 125, 75, 50, 25, 0])
        grid2[grid_indx].set_yticklabels([])
        grid2[grid_indx].set_xticklabels([])
        grid2[grid_indx].get_yaxis().set_tick_params(direction='out', width=1, length=2, top='off')
        grid2[grid_indx].get_xaxis().set_tick_params(direction='out', width=1, length=2, top='off')
        grid2[grid_indx].set_ylim([200, 125])
        grid2[grid_indx].set_xlim([25, 175])

        # Plot pressure for particular scenario and interval
        im3 = grid3[grid_indx].imshow(avg_pressure.T, interpolation='bilinear', cmap=cmap3, vmin=0, vmax=1.8)
        if j==1:
            grid3[grid_indx].set_ylabel(titles[i], fontsize=11, position=(1,0.5), verticalalignment='bottom')
        if grid_indx in {0, 1, 2, 3}:
            grid3[grid_indx].set_title(interval_titles[j-1], fontsize=11)
        grid3[grid_indx].set_yticks([200, 175, 150, 125, 100, 75, 50, 25, 0])
        grid3[grid_indx].set_xticks([200, 175, 150, 125, 75, 50, 25, 0])
        grid3[grid_indx].set_yticklabels([])
        grid3[grid_indx].set_xticklabels([])
        grid3[grid_indx].get_yaxis().set_tick_params(direction='out', width=1, length=2, top='off')
        grid3[grid_indx].get_xaxis().set_tick_params(direction='out', width=1, length=2, top='off')
        grid3[grid_indx].set_ylim([200, 125])
        grid3[grid_indx].set_xlim([25, 175])

        # Plot radial pressure for particular scenario and interval
        im4 = grid4[grid_indx].imshow(avg_radial_pressure.T, interpolation='bilinear', cmap=cmap4, vmin=0, vmax=1.8)
        if j==1:
            grid4[grid_indx].set_ylabel(titles[i], fontsize=11, position=(1,0.5), verticalalignment='bottom')
        if grid_indx in {0, 1, 2, 3}:
            grid4[grid_indx].set_title(interval_titles[j-1], fontsize=11)
        grid4[grid_indx].set_yticks([200, 175, 150, 125, 100, 75, 50, 25, 0])
        grid4[grid_indx].set_xticks([200, 175, 150, 125, 75, 50, 25, 0])
        grid4[grid_indx].set_yticklabels([])
        grid4[grid_indx].set_xticklabels([])
        grid4[grid_indx].get_yaxis().set_tick_params(direction='out', width=1, length=2, top='off')
        grid4[grid_indx].get_xaxis().set_tick_params(direction='out', width=1, length=2, top='off')
        grid4[grid_indx].set_ylim([200, 125])
        grid4[grid_indx].set_xlim([25, 175])

# Plot colorbars to the figures
for ci in range(0,n_scenarios):
    cbar0 = grid0.cbar_axes[ci].colorbar(im0, ticks=[0,0.7,1.4], )
    cbar1 = grid1.cbar_axes[ci].colorbar(im1, ticks=[0,0.75,1.5,2.25,3], )
    cbar2 = grid2.cbar_axes[ci].colorbar(im2, ticks=[0,0.25,0.5,0.75,1], )
    cbar3 = grid3.cbar_axes[ci].colorbar(im3, ticks=[0,0.6,1.2,1.8], )
    cbar4 = grid4.cbar_axes[ci].colorbar(im4, ticks=[0,0.6,1.2,1.8], )
    cbar0.ax.tick_params(labelsize=11)
    cbar1.ax.tick_params(labelsize=11)
    cbar2.ax.tick_params(labelsize=11)
    cbar3.ax.tick_params(labelsize=11)
    cbar4.ax.tick_params(labelsize=11)

    cbar0.ax.set_ylabel(r'Speed (m/s)', rotation=90)
    cbar1.ax.set_ylabel(r'Density (1/$m^2$)', rotation=90)
    cbar2.ax.set_ylabel(r'Speed (m/s)', rotation=90)
    cbar3.ax.set_ylabel(r'Crowd pressure (1/s${}^2$)', rotation=90)
    cbar4.ax.set_ylabel(r'Crowd pressure (1/s${}^2$)', rotation=90)


plt.figure(0)
plt.savefig('temporal_evolution_speed.pdf'
            , bbox_inches='tight'
           )

plt.figure(1)
plt.savefig('temporal_evolution_density.pdf'
            , bbox_inches='tight'
           )

plt.figure(2)
plt.savefig('temporal_evolution_radial_speed.pdf'
            , bbox_inches='tight'
           )

plt.figure(3)
plt.savefig('temporal_evolution_pressure.pdf'
            , bbox_inches='tight'
           )

plt.figure(4)
plt.savefig('temporal_evolution_radial_pressure.pdf'
            , bbox_inches='tight'
           )

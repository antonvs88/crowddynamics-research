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
from radial_mean import radial_mean

# Cell size in the grid, determines the resolution of the micro-macro converted data (this was determined in "calculate_field_data.py")
width = 20
height = 20
cell_size = 0.1
m = np.round(width/cell_size) # number of cells in horizontal direction
n = np.round(height/cell_size) # number of cells in vertical direction
m = m.astype(int)
n = n.astype(int)

# Names of folders containing the field data averaged over time and sample for different scenarios.
mylist = ['taset0_normi', 'taset80_normi', 'taset150_normi', 'taset220_normi', 'taset350_normi', 'taset500_normi',
          'vd1_A2000_normi', 'vd2_A1750_normi', 'vd3_A1500_normi', 'vd4_A1250_normi']

# Figure titles
titles = [r'$T_{ASET}$=0', r'$T_{ASET}$=80', r'$T_{ASET}$=150', r'$T_{ASET}$=220', r'$T_{ASET}$=350', r'$T_{ASET}$=500',
          r'$v_{des}$ = 1 m/s', r'$v_{des}$ = 2 m/s', r'$v_{des}$ = 3 m/s', r'$v_{des}$ = 4 m/s']

chosen = [0,1,2,3,4,5,6,7,8,9] # scenarios chosen from "mylist"
n_chosen = len(chosen) # number of scenarios
games = chosen[:-4] # scenarios where the game is played
n_games = len(games) # number of scenarios where the game is played

# Arrays for the averaged field data
speed = np.zeros((n_chosen, n, m))
crowd_pressure = np.zeros((n_chosen, n, m))

# Initialize figure
fig = plt.figure(0)
grid_cell = 12
G = gridspec.GridSpec(2*grid_cell, 2*grid_cell)
ax0 = plt.subplot(G[1:grid_cell-1, 2:grid_cell-1])
ax1 = plt.subplot(G[3:grid_cell-1, grid_cell:2*grid_cell])
ax2 = plt.subplot(G[grid_cell + 3:2*grid_cell, 2:grid_cell-1])
ax3 = plt.subplot(G[grid_cell + 3:2*grid_cell, grid_cell+2:2*grid_cell - 1])

# Colormap used to plot average radial "crowd pressure" field
cmap = plt.cm.inferno

# Loop over all scenarios
for i in range(0,n_chosen):

    # Open averaged speed field data for particular scenario and interval
    with h5py.File("{}{}{}".format('average_fields/', mylist[i], '/average_speed_interval1.hdf5'),
                   'r') as hf1:
        speed[i,:,:] = hf1['average_fields'][mylist[i]]['average_speed_interval1.npy.gz'][()]

    # Open averaged radial "crowd pressure" field data for particular scenario and interval
    with h5py.File("{}{}{}".format('average_fields/', mylist[i], '/average_radial_pressure_interval1.hdf5'),
                   'r') as hf2:
        crowd_pressure[i,:,:] = hf2['average_fields'][mylist[i]]['average_radial_pressure_interval1.npy.gz'][()]

    # Calculate distance vs. avg_speed and distance vs. avg_pressure data
    distances, avg_speed, avg_crowd_pressure = radial_mean(speed[i,:,:], crowd_pressure[i,:,:], cell_size, width, height)

    # Plot distance vs. speed (both for scenarios with and without game) in the first grid cell in the plot.
    if i in range(0,n_games):
        ax0.plot(distances, avg_speed, label='_nolegend_', color='red', linewidth=1.5)
    if i in range(n_games,n_chosen):
        ax0.plot(distances, avg_speed, label='_nolegend_', color='black', linewidth=1.5)
    ax0.set_ylim([0, 1.2])
    ax0.set_xlim([0, 8])
    ax0.set_xlabel(r'Radial distance from exit (m)', fontsize=13)
    ax0.set_ylabel(r'Speed (m/s)', fontsize=13)
    ax0.tick_params(axis='both', which='major', labelsize=12)
    ax0.get_yaxis().set_tick_params(direction='out', width=2, top='off',pad=0)
    ax0.get_xaxis().set_tick_params(direction='out', width=2, top='off',pad=0)
    ax0.set_yticks([0,0.7,1.4])
    ax0.set_xticks([0,2.5,5,7.5])
    ax0.yaxis.set_label_coords(-0.15,0.5)
    ax0.xaxis.set_label_coords(0.5,-0.2)

    # Make a customized legend for the first grid cell in the plot.
    if i == (n_chosen - 1):
        ax0.text(1.7, 1.15, r'$T_{ASET}$', fontsize=12)
        ax0.text(1.7, 1, r'$v^0$', fontsize=13)
        ax0.text(0.84, 0.93, r'                   =500, 350, 220, 150, 80, 0' + '\n' +
                 r'             =1, 2, 3, 4 (m/s)' + '\n' + 'From top to bottom along the dotted lines',
                 linespacing=1.5,
                 bbox=dict(facecolor='none', edgecolor='black', pad=2), fontsize=9)

        x_top = 1.25
        y_top = 0.87
        x = x_top*np.ones(2)
        y = [0, y_top]
        ax0.plot(x, y, linestyle='--', color='black', linewidth=1.5)
        ax0.arrow(x_top, 0.87, 0, -0.03, head_width=0.15, head_length=0.05, fc='k', ec='k', width=0.03)

        x_top_red = 6.7
        y_top_red = 0.87
        x_red = x_top_red*np.ones(2)
        y_red = [0, y_top_red]
        ax0.plot(x_red, y_red, linestyle='--', color='red', linewidth=1.5)
        ax0.arrow(x_top_red, 0.87, 0, -0.03, head_width=0.15, head_length=0.05, fc='red', ec='red', width=0.03)

        ax0.add_patch(ptch.Rectangle((0.96, 1.07), 0.6, 0.017, edgecolor='black', facecolor='black'),)
        ax0.add_patch(ptch.Rectangle((0.96,1.18), 0.6, 0.017, edgecolor='red', facecolor='red'),)

    # Plot averaged radial "crowd pressure" field for Taset=500 in the second grid cell in the plot.
    if i == (n_games - 1):
        im = ax1.imshow(crowd_pressure[i, :, :].T, interpolation='bilinear', cmap=cmap, vmin=0, vmax=1.8)
        ax1.set_yticks([200, 175, 150, 125, 100, 75, 50, 25, 0])
        ax1.set_xticks([200, 175, 150, 125, 75, 50, 25, 0])
        ax1.set_yticklabels([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20], fontsize=12)
        ax1.set_xticklabels([10, 7.5, 5, 2.5, -2.5, -5, -7.5, -10], fontsize=12)
        ax1.get_yaxis().set_tick_params(direction='out', width=2, top='off')
        ax1.get_xaxis().set_tick_params(direction='out', width=2, top='off')
        ax1.set_ylim([198, 123])
        ax1.set_xlim([25, 175])
        ax1.xaxis.set_label_coords(-0.1, -0.05)
        cax = fig.add_axes([0.555, 0.81, 0.39, 0.03])
        cbar = fig.colorbar(im, orientation='horizontal', cax=cax, ticks=[0, 0.6, 1.2, 1.8],
                            label=r'Crowd pressure (1/$s^2$)')
        ax = plt.gca()
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.tick_params(labelsize=12, direction='out', width=2)
        cbar.ax.xaxis.label.set_size(13)

        x = [200 * 37 / 80, 200 * 42 / 80]
        y = 200 * np.ones(2)
        ax1.plot(x, y, color='black', linewidth=5, clip_on=False)
        ax1.text(200 * 34.5 / 80, 200 * 85 / 80, 'EXIT', fontweight='bold', fontsize=16)
        ax1.text(200 * 11 / 80, 200 * 88.25 / 80, r'Horizontal distance from exit (m)', fontsize=13)
        ax1.text(-200 * 1.4 / 80, 200 * 41 / 80, r'Vertical distance from exit (m)', fontsize=13, rotation='vertical')

    # Plot distance vs. radial "crowd pressure" (for scenarios with game) in the third grid cell in the plot.
    if i in range(0,n_games):
        ax2.plot(distances, avg_crowd_pressure, label='_nolegend_', color='red', linewidth=1.5)

    ax2.set_ylim([0, 0.4])
    ax2.set_xlim([0, 8])
    ax2.set_xlabel(r'Radial distance from exit (m)', fontsize=13)
    ax2.set_ylabel(r'Crowd pressure (1/($s^2$))', fontsize=13)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.get_yaxis().set_tick_params(direction='out', width=2, top='off', pad=0)
    ax2.get_xaxis().set_tick_params(direction='out', width=2, top='off', pad=0)
    ax2.set_yticks([0,0.6,1.2,1.8])
    ax2.set_xticks([0,2.5,5,7.5])
    ax2.yaxis.set_label_coords(-0.15,0.37)
    ax2.xaxis.set_label_coords(0.5,-0.2)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # Make a customized legend for the third grid cell in the plot.
    if i == (n_chosen - 1):

        offset=0.4
        x_top = 5.65
        y_top = 0.8+offset
        x = x_top*np.ones(2)
        y = [0, y_top]
        ax2.plot(x, y, linestyle='--', color='red', linewidth=1.5)
        ax2.text(1.57, 1.02+offset, r'$T_{ASET}$', fontsize=12)
        ax2.text(0.8, 0.87 + offset, r'                   =500, 350, 220, 150, 80, 0' + '\n'
                 + 'From top to bottom along the dotted line',
                 linespacing=1.5,
                 bbox=dict(facecolor='none', edgecolor='black', pad=2), fontsize=9)
        ax2.add_patch(ptch.Rectangle((0.88,1.07 + offset), 0.56, 0.03, edgecolor='red', facecolor='red'),)
        ax2.arrow(x_top, 1.16, 0, -0.01, head_width=0.15, head_length=0.1, fc='r', ec='r')

    # Plot distance vs. radial "crowd pressure" (for scenarios without game) in the fourth grid cell in the plot.
    if i in range(n_games, n_chosen):
        ax3.plot(distances, avg_crowd_pressure, label='_nolegend_', color='black', linewidth=1.5)

    ax3.set_ylim([0, 0.4])
    ax3.set_xlim([0, 8])
    ax3.set_xlabel(r'Radial distance from exit (m)', fontsize=13)
    ax3.set_ylabel(r'Crowd pressure (1/($s^2$))', fontsize=13)
    ax3.tick_params(axis='both', which='major', labelsize=12)
    ax3.get_yaxis().set_tick_params(direction='out', width=2, top='off', pad=0)
    ax3.get_xaxis().set_tick_params(direction='out', width=2, top='off', pad=0)
    ax3.set_yticks([0, 0.6, 1.2, 1.8])
    ax3.set_xticks([0, 2.5, 5, 7.5])
    ax3.yaxis.set_label_coords(-0.15, 0.37)
    ax3.xaxis.set_label_coords(0.5, -0.2)
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # Make a customized legend for the fourth grid cell in the plot.
    if i == (n_chosen - 1):

        offset = 0.4
        x_top = 1.05
        y_top = 0.82 + offset
        x = x_top * np.ones(2)
        y = [0, y_top]
        ax3.plot(x, y, linestyle='--', color='black', linewidth=1.5)
        ax3.text(1.57, 1.02 + offset, r'$v^0$', fontsize=12)
        ax3.text(0.8, 0.9 + offset, r'             =1, 2, 3, 4 (m/s)' + '\n'
                 + 'From top to bottom along the dotted line',
                 linespacing=1.5,
                 bbox=dict(facecolor='none', edgecolor='black', pad=2), fontsize=9)
        ax3.add_patch(ptch.Rectangle((0.88, 1.08 + offset), 0.56, 0.03, edgecolor='black', facecolor='black'), )
        ax3.arrow(x_top, 1.18, 0, -0.01, head_width=0.15, head_length=0.1, fc='k', ec='k')

# Label the plots in the grid cells with letters from a to d.
fig.text(0.001, 0.88, 'a', fontweight='bold', fontsize=16, transform=fig.transFigure)
fig.text(0.455, 0.89, 'b', fontweight='bold', fontsize=16, transform=fig.transFigure)
fig.text(0.001, 0.41, 'c', fontweight='bold', fontsize=16, transform=fig.transFigure)
fig.text(0.47, 0.39, 'd', fontweight='bold', fontsize=16, transform=fig.transFigure)

G.update(left=0, right=1, wspace=0) # grid settings
plt.savefig('figure_4.pdf'
            ,bbox_inches='tight'
           )

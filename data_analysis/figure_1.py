import numpy as np
import h5py
import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
from matplotlib.artist import setp
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
import os
from shapely.geometry import Point, Polygon, LineString
from mpl_toolkits.axes_grid1 import ImageGrid

# Room dimensions
width = 20
height = 20

# Names of folders containing the strategy data for different scenarios.
mylist = ['taset0', 'taset80', 'taset150', 'taset500']

# Figure titles
titles = [r'$T_{ASET}$=0', r'$T_{ASET}$=80', r'$T_{ASET}$=150', r'$T_{ASET}$=500']

# Initialize figure
fig = plt.figure(0)
grid = ImageGrid(fig, 111, nrows_ncols=(2,2), axes_pad=0.35, aspect=True)

# Create a color map of fixed colors
cmap = colors.ListedColormap(['red', 'blue'])
bounds=[0,0.5,1]
norm = colors.BoundaryNorm(bounds, cmap.N)

# The simulation which data is used to plot the equilibria (qualitatively the equilibria for different
# simulations look the same.
j = 99

# Loop through the different scenarios
for i in range(0,4):

    # Load data of which pedestrians are in the room
    if os.path.exists("{}{}{}{}{}".format(mylist[i], '/', 'in_room1', j, '.npy.gz')):
        in_room = np.loadtxt("{}{}{}{}{}".format(mylist[i], '/', 'in_room1', j, '.npy.gz'))
        sum_in_room = np.sum(in_room, axis=1)

        # Time when 10 pedestrians have exited
        time_stat_reg_start = np.where(sum_in_room == 190)[0][0]

        # Pedestrians in room when 10 pedestrians have exited
        agents_in_room = np.where(in_room[time_stat_reg_start, :] == 1)[0]

    # Load pedestrian's x-positions
    if os.path.exists("{}{}{}{}{}".format(mylist[i], '/', 'positions_x', j, '.npy.gz')):
        positions_x = np.loadtxt("{}{}{}{}{}".format(mylist[i], '/', 'positions_x', j, '.npy.gz'))
        positions_x = positions_x[0::2]

        # Which pedestrians have exited the room
        outside_room = np.where(in_room[time_stat_reg_start, :] == 0)[0]

    # Load pedestrian's y-positions
    if os.path.exists("{}{}{}{}{}".format(mylist[i], '/', 'positions_y', j, '.npy.gz')):
        positions_y = np.loadtxt("{}{}{}{}{}".format(mylist[i], '/', 'positions_y', j, '.npy.gz'))
        positions_y = positions_y[0::2]

    # Load pedestrian's radii
    if os.path.exists("{}{}{}{}{}".format(mylist[i], '/', 'radius', j, '.npy.gz')):
        radius = np.loadtxt("{}{}{}{}{}".format(mylist[i], '/', 'radius', j, '.npy.gz'))

    # Load pedestrian's strategies
    if os.path.exists("{}{}{}{}{}".format(mylist[i], '/', 'strategy', j, '.npy.gz')):
        strategy = np.loadtxt("{}{}{}{}{}".format(mylist[i], '/', 'strategy', j, '.npy.gz'))

    # Create circles based on pedestrian's positions and radius
    patches = []
    for k in agents_in_room:
        circle = ptch.Circle((positions_y[time_stat_reg_start, k], -positions_x[time_stat_reg_start, k] + width),
                              radius[k])
        patches.append(circle)

    if i == 0:

        # Add legend
        grid[i].add_patch(ptch.Circle((2.5,8.3), 0.25, edgecolor='black', facecolor='red', lw=0.2),)
        grid[i].add_patch(ptch.Circle((2.5,7.3), 0.25, edgecolor='black', facecolor='blue', lw=0.2),)
        grid[i].text(3, 8, 'Impatient', fontsize=12)
        grid[i].text(3, 7, 'Patient', fontsize=12)

        # Change figure settings
        grid[i].set_xlim([2, 18])
        grid[i].set_ylim([-2, 9])
        grid[i].get_xaxis().set_visible(False)
        grid[i].get_yaxis().set_visible(False)
        grid[i].set_aspect('equal')

        # Add colored circles to represent different strategists
        p = PatchCollection(patches, cmap=cmap, norm=norm, edgecolor='black', lw=0.2)
        p.set_array(strategy[time_stat_reg_start, agents_in_room])
        grid[i].add_collection(p)

        # Set Taset parameter as figure title
        grid[i].set_title(titles[i], fontsize=12, y=1)

        # Set EXIT sign
        grid[i].text(8.8,-1.7,'EXIT', fontweight='bold', fontsize=16)

        # Plot a "bottom floor"
        floor_left_x = np.arange(0,10,0.5)
        floor_left_y = -0.04*np.ones(20)
        floor_right_x = np.arange(11,21,1)
        floor_right_y = -0.04*np.ones(10)
        grid[i].plot(floor_left_x, floor_left_y, color='black', linewidth=2)
        grid[i].plot(floor_right_x, floor_right_y, color='black', linewidth=2)

        # Figure letter
        grid[i].text(2.1, 9.5, 'a', fontweight='bold', fontsize=14)

    if i == 1:

        # Add legend
        grid[i].add_patch(ptch.Circle((2.5,8.3), 0.25, edgecolor='black', facecolor='red', lw=0.2),)
        grid[i].add_patch(ptch.Circle((2.5,7.3), 0.25, edgecolor='black', facecolor='blue', lw=0.2),)
        grid[i].text(3, 8, 'Impatient', fontsize=12)
        grid[i].text(3, 7, 'Patient', fontsize=12)

        # Change figure settings
        grid[i].set_xlim([2, 18])
        grid[i].set_ylim([-2, 9])
        grid[i].get_xaxis().set_visible(False)
        grid[i].get_yaxis().set_visible(False)
        grid[i].set_aspect('equal')

        # Add colored circles to represent different strategists
        p = PatchCollection(patches, cmap=cmap, norm=norm, edgecolor='black', lw=0.2)
        p.set_array(strategy[time_stat_reg_start, agents_in_room])
        grid[i].add_collection(p)

        # Set Taset parameter as figure title
        grid[i].set_title(titles[i], fontsize=12, y=1)

        # Set EXIT sign
        grid[i].text(8.8,-1.7,'EXIT', fontweight='bold', fontsize=16)
        
        # Plot a "bottom floor"
        floor_left_x = np.arange(0,10,0.5)
        floor_left_y = -0.04*np.ones(20)
        floor_right_x = np.arange(11,21,1)
        floor_right_y = -0.04*np.ones(10)
        grid[i].plot(floor_left_x, floor_left_y, color='black', linewidth=2)
        grid[i].plot(floor_right_x, floor_right_y, color='black', linewidth=2)

        # Figure letter
        grid[i].text(2.1, 9.5, 'b', fontweight='bold', fontsize=14)

    if i == 2:

        # Add legend
        grid[i].add_patch(ptch.Circle((2.5,8.3), 0.25, edgecolor='black', facecolor='red', lw=0.2),)
        grid[i].add_patch(ptch.Circle((2.5,7.3), 0.25, edgecolor='black', facecolor='blue', lw=0.2),)
        grid[i].text(3, 8, 'Impatient', fontsize=12)
        grid[i].text(3, 7, 'Patient', fontsize=12)

        # Change figure settings
        grid[i].set_xlim([2, 18])
        grid[i].set_ylim([-2, 9])
        grid[i].get_xaxis().set_visible(False)
        grid[i].get_yaxis().set_visible(False)
        grid[i].set_aspect('equal')

        # Add colored circles to represent different strategists
        p = PatchCollection(patches, cmap=cmap, norm=norm, edgecolor='black', lw=0.2)
        p.set_array(strategy[time_stat_reg_start, agents_in_room])
        grid[i].add_collection(p)

        # Set Taset parameter as figure title
        grid[i].set_title(titles[i], fontsize=12, y=1)

        # Set EXIT sign
        grid[i].text(8.8,-1.7,'EXIT', fontweight='bold', fontsize=16)

        # Plot a "bottom floor"
        floor_left_x = np.arange(0,10,0.5)
        floor_left_y = -0.04*np.ones(20)
        floor_right_x = np.arange(11,21,1)
        floor_right_y = -0.04*np.ones(10)
        grid[i].plot(floor_left_x, floor_left_y, color='black', linewidth=2)
        grid[i].plot(floor_right_x, floor_right_y, color='black', linewidth=2)

        # Figure letter
        grid[i].text(2.1, 9.5, 'c', fontweight='bold', fontsize=14)

    if i == 3:

        # Add legend
        grid[i].add_patch(ptch.Circle((2.5,8.3), 0.25, edgecolor='black', facecolor='red', lw=0.2),)
        grid[i].add_patch(ptch.Circle((2.5,7.3), 0.25, edgecolor='black', facecolor='blue', lw=0.2),)
        grid[i].text(3, 8, 'Impatient', fontsize=12)
        grid[i].text(3, 7, 'Patient', fontsize=12)

        # Change figure settings
        grid[i].set_xlim([2, 18])
        grid[i].set_ylim([-2, 9])
        grid[i].get_xaxis().set_visible(False)
        grid[i].get_yaxis().set_visible(False)
        grid[i].set_aspect('equal')

        # Add colored circles to represent different strategists
        p = PatchCollection(patches, cmap=cmap, norm=norm, edgecolor='black', lw=0.2)
        p.set_array(strategy[time_stat_reg_start, agents_in_room])
        grid[i].add_collection(p)

        # Set Taset parameter as figure title
        grid[i].set_title(titles[i], fontsize=12, y=1)

        # Set EXIT sign
        grid[i].text(8.8,-1.7,'EXIT', fontweight='bold', fontsize=16)

        # Plot a "bottom floor"
        floor_left_x = np.arange(0,10,0.5)
        floor_left_y = -0.04*np.ones(20)
        floor_right_x = np.arange(11,21,1)
        floor_right_y = -0.04*np.ones(10)
        grid[i].plot(floor_left_x, floor_left_y, color='black', linewidth=2)
        grid[i].plot(floor_right_x, floor_right_y, color='black', linewidth=2)

        # Figure letter
        grid[i].text(2.1, 9.5, 'd', fontweight='bold', fontsize=14)

# Add a black rectangle in each figure to represent the exit
grid[0].add_patch(ptch.Rectangle((9.4,-0.41), 1.6, 0.4, edgecolor='black', facecolor='black'),)
grid[1].add_patch(ptch.Rectangle((9.4,-0.41), 1.6, 0.4, edgecolor='black', facecolor='black'),)
grid[2].add_patch(ptch.Rectangle((9.4,-0.41), 1.6, 0.4, edgecolor='black', facecolor='black'),)
grid[3].add_patch(ptch.Rectangle((9.4,-0.41), 1.6, 0.4, edgecolor='black', facecolor='black'),)

# Save figure as pdf
plt.savefig('figure_1.pdf',
            bbox_inches='tight'
            )

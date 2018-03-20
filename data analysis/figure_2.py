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

# Number of pedestrians in room initially
n_a = 1000

# Number of pedestrians exited
stat_reg_start = 9

# Initialize figure
fig = plt.figure(0)

# Create a color map of fixed colors
cmap = colors.ListedColormap(['red', 'blue'])
bounds=[0,0.5,1]
norm = colors.BoundaryNorm(bounds, cmap.N)

# Load data of which pedestrians are in the room
if os.path.exists('bigequilibrium/in_room1.npy.gz'):
    in_room = np.loadtxt('bigequilibrium/in_room1.npy.gz')
    sum_in_room = np.sum(in_room, axis=1)

    # Time when n_a - stat_reg_start pedestrians have exited
    time_stat_reg_start = np.where(sum_in_room == (n_a -stat_reg_start))[0][0]
    agents_in_room = np.where(in_room[time_stat_reg_start, :] == 1)[0]

# Load pedestrian's x-positions
if os.path.exists('bigequilibrium/positions_x.npy.gz'):
    positions_x = np.loadtxt('bigequilibrium/positions_x.npy.gz')
    positions_x = positions_x[0::2]

# Load pedestrian's y-positions
if os.path.exists('bigequilibrium/positions_y.npy.gz'):
    positions_y = np.loadtxt('bigequilibrium/positions_y.npy.gz')
    positions_y = positions_y[0::2]

# Load pedestrian's radii
if os.path.exists('bigequilibrium/radius.npy.gz'):
    radius = np.loadtxt('bigequilibrium/radius.npy.gz')

# Load pedestrian's strategies
if os.path.exists('bigequilibrium/strategy.npy.gz'):
    strategy = np.loadtxt('bigequilibrium/strategy.npy.gz')

# Create cricles based on pedestrian's positions and radius
patches = []
for k in agents_in_room:
    circle = ptch.Circle((positions_y[time_stat_reg_start, k], -positions_x[time_stat_reg_start, k] + width),
                              radius[k])
    patches.append(circle)

# Change figure settings
ax = plt.gca()
ax.set_xlim([5.53, 34.43])
ax.set_ylim([-2, 14.10])
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.set_aspect('equal')

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Add colored circles to represent different strategists
p = PatchCollection(patches, cmap=cmap, norm=norm, edgecolor='black', lw=0.2)
p.set_array(strategy[time_stat_reg_start, agents_in_room])
ax.add_collection(p)

# Add legend
ax.add_patch(ptch.Circle((6.1,13.5), 0.5, edgecolor='black', facecolor='red', lw=0.2),)
ax.add_patch(ptch.Circle((6.1,12.1), 0.5, edgecolor='black', facecolor='blue', lw=0.2),)
ax.text(7.1, 13.1, 'Impatient', fontsize=20)
ax.text(7.1, 11.7, 'Patient', fontsize=20)


# Plot a "bottom floor"
floor_left_x = np.arange(-1,19.8,0.1)
floor_left_y = -0.08*np.ones(208)
floor_right_x = np.arange(21.1,42,0.1)
floor_right_y = -0.08*np.ones(209)
plt.plot(floor_left_x, floor_left_y, color='black', linewidth=2.5)
plt.plot(floor_right_x, floor_right_y, color='black', linewidth=2.5)

# Plot 3 half-circles
x0 = 20.3
y0 = 0
radius0 = 9
x = np.arange(x0-radius0,x0+radius0,0.001)
y = np.sqrt(radius0**2-(x-x0)**2) + y0
plt.plot(x, y, color='black', linewidth=5)

x1 = 20.3
y1 = 0
radius1 = 6.5
x = np.arange(x1-radius1,x1+radius1,0.001)
y = np.sqrt(radius1**2-(x-x1)**2) + y1
plt.plot(x, y, color='black', linewidth=5)

x2 = 20.3
y2 = 0
radius2 = 3.5
x = np.arange(x2-radius2,x2+radius2,0.001)
y = np.sqrt(radius2**2-(x-x2)**2) + y2
plt.plot(x, y, color='black', linewidth=5)

# Plot black rectangle to represent the exit
ax.add_patch(ptch.Rectangle((19.3,-0.7), 2.4,0.7, edgecolor='black', facecolor='black'),)

# Plot EXIT sign
plt.text(18.85, -2, 'EXIT', fontsize=20, fontweight='bold')

# Label the half-circles
plt.text(x0-radius0-0.5, -1.4, 'A', fontsize=20, fontweight='bold')
plt.text(x1-radius1-0.5, -1.4, 'B', fontsize=20, fontweight='bold')
plt.text(x2-radius2-0.5, -1.4, 'C', fontsize=20, fontweight='bold')

# Save figure as pdf
plt.savefig('figure_2.pdf',
            bbox_inches='tight'
            )

from scipy.spatial import Voronoi, voronoi_plot_2d

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from shapely.geometry import Polygon, MultiLineString, Point
from shapely.ops import polygonize
from descartes import PolygonPatch
from voronoi_finite_polygons_2d import voronoi_finite_polygons_2d
from recursive_mean import recursive_mean

# The data is divided into four intervals.
n_intervals = 4
# The number of simulations.
n_simulations = 50

# Cell size in the grid, determines the resolution of the micro-macro converted data (this was determined in "calculate_field_data.py")
cell_size = 0.1
width = 20
height = 20
m = np.round(width / cell_size) # number of cells in horizontal direction
n = np.round(height / cell_size) # number of cells in vertical direction
m = m.astype(int)
n = n.astype(int)

# We will calculate mean values of some part of the data recursively by taking a "chunk" of the data.
chunk = 1000 # chunk size

# Name of the folder, where the data is; can be an array of folders
mylist = ['vd4_A1250_normi']

# The outer loop goes through the intervals (the data is divided in four time intervals, where first interval contains data when there are between 190 to 145 pedestrians in the room, second interval when there are 145 to 100, third when there are between 100 and 55, and fourth when there are between 55 to 10).
for i in range(1,n_intervals+1):

    # For each simulation, store the number of time steps in the interval in an array.
    time_steps = np.zeros(n_simulations+1)
    for j in range(0,n_simulations):
        intervals = np.load("{}{}{}{}{}{}".format('fields/', mylist[0], '/', 'intervals', j, '.npy'))
        time_steps[j+1] = intervals[i-1]

    # Calculate cumulative sum of the time steps in the interval, and store it in an array.
    time_steps = time_steps.astype(int)
    time_steps = np.cumsum(time_steps)

    # Arrays for storing the spatiotemporal field data. The data stored has been produced with the "calculate_field_data.py" function.
    spds = np.empty((time_steps[-1], n, m), dtype=np.float16)
    prjs = np.empty(spds.shape, dtype=np.float16)
    dnsts = np.empty(spds.shape, dtype=np.float16)
    # Loop through the simulations
    for j in range(0,n_simulations):

        with h5py.File("{}{}{}{}{}{}{}".format('fields/', mylist[0], '/', 'speed_interval', i, j, '.hdf5'),
                       'r') as hf1:
            speed = hf1['fields'][mylist[0]]["{}{}{}{}".format('speed_interval', i, j, '.npy.gz')][()]

        with h5py.File("{}{}{}{}{}{}{}".format('fields/', mylist[0], '/', 'density_interval', i, j, '.hdf5'),
                       'r') as hf2:
            density = hf2['fields'][mylist[0]]["{}{}{}{}".format('density_interval', i, j, '.npy.gz')][()]

        with h5py.File("{}{}{}{}{}{}{}".format('fields/', mylist[0], '/', 'projection_interval', i, j, '.hdf5'),
                       'r') as hf3:
            projection = hf3['fields'][mylist[0]]["{}{}{}{}".format('projection_interval', i, j, '.npy.gz')][()]

        spds[time_steps[j]:time_steps[j+1], :, :] = speed
        dnsts[time_steps[j]:time_steps[j+1], :, :] = density
        prjs[time_steps[j]:time_steps[j+1], :, :] = projection

    # Calculate mean speed (over time and samples) recursively
    speed_time_sample_average = recursive_mean(spds, chunk)

    # Calculate mean density (over time and samples) recursively
    density_time_sample_average = recursive_mean(dnsts, chunk)

    # Calculate mean radial speed (over time and samples) recursively
    projection_time_sample_average = recursive_mean(prjs, chunk)

    # Speed variance (over time and samples)
    speed_time_sample_dvs = spds - speed_time_sample_average
    speed_time_sample_dvs = speed_time_sample_dvs.astype(np.float16)
    speed_time_sample_dvs *= speed_time_sample_dvs
    speed_time_sample_variance = recursive_mean(speed_time_sample_dvs, chunk)

    # Radial speed variance (over time and samples)
    projection_time_sample_dvs = prjs - projection_time_sample_average
    projection_time_sample_dvs = projection_time_sample_dvs.astype(np.float16)
    projection_time_sample_dvs *= projection_time_sample_dvs
    projection_time_sample_variance = recursive_mean(projection_time_sample_dvs, chunk)

    # Average "crowd pressure" (over time and samples)
    pressure_time_sample_average = density_time_sample_average * speed_time_sample_variance
    
    # Average radial "crowd pressure" (over time and samples)
    pressure_radial_time_sample_average = density_time_sample_average * projection_time_sample_variance

    # Save averaged speed, density, radial speed, "crowd pressure", and radial "crowd pressure" fields
    with h5py.File("{}{}{}{}{}{}".format('average_fields/', mylist[0], '/', 'average_speed_interval', i, '.hdf5'), 'w') as xxx1:
        xxx1.create_dataset("{}{}{}{}{}{}".format('average_fields/', mylist[0], '/', 'average_speed_interval', i, '.npy.gz'),
                           data=speed_time_sample_average)

    with h5py.File("{}{}{}{}{}{}".format('average_fields/', mylist[0], '/', 'average_density_interval', i, '.hdf5'), 'w') as xxx2:
        xxx2.create_dataset("{}{}{}{}{}{}".format('average_fields/', mylist[0], '/', 'average_density_interval', i, '.npy.gz'),
                           data=density_time_sample_average)

    with h5py.File("{}{}{}{}{}{}".format('average_fields/', mylist[0], '/', 'average_radial_speed_interval', i, '.hdf5'), 'w') as xxx3:
        xxx3.create_dataset("{}{}{}{}{}{}".format('average_fields/', mylist[0], '/', 'average_radial_speed_interval', i, '.npy.gz'),
                           data=projection_time_sample_average)

    with h5py.File("{}{}{}{}{}{}".format('average_fields/', mylist[0], '/', 'average_pressure_interval', i, '.hdf5'), 'w') as xxx4:
        xxx4.create_dataset("{}{}{}{}{}{}".format('average_fields/', mylist[0], '/', 'average_pressure_interval', i, '.npy.gz'),
                           data=pressure_time_sample_average)

    with h5py.File("{}{}{}{}{}{}".format('average_fields/', mylist[0], '/', 'average_radial_pressure_interval', i, '.hdf5'), 'w') as xxx5:
        xxx5.create_dataset("{}{}{}{}{}{}".format('average_fields/', mylist[0], '/', 'average_radial_pressure_interval', i, '.npy.gz'),
                           data=pressure_radial_time_sample_average)

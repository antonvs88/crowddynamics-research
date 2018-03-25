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

# Bound box representing the room. Used later in making Voronoi tessalation.
width = 20
height = 20
boundbox = Polygon([(0, 0), (0, height), (width, height), (width, 0)])

# Create a grid structure over the room geometry.
# Cell size in the grid, determines the resolution of the micro-macro converted data
cell_size = 0.1
m = np.round(width / cell_size)
n = np.round(height / cell_size)
m = m.astype(int)
n = n.astype(int)
X = np.linspace(0, width, m + 1)
Y = np.linspace(0, height, n + 1)
hlines = [((x1, yi), (x2, yi)) for x1, x2 in zip(X[:-1], Y[1:]) for yi in Y]
vlines = [((xi, y1), (xi, y2)) for y1, y2 in zip(Y[:-1], Y[1:]) for xi in X]
grids = list(polygonize(MultiLineString(hlines + vlines)))

# The data is divided into four intervals. The number of pedestrians in the room determines the intervals.
# The data when the 10 first and 10 last pedestrians leave the room is omitted to get rid of transient
# behavior of the "crowd system".
interval1_start = 190
interval2_start = 145
interval3_start = 100
interval4_start = 55
interval4_end = 10

# These should be the midpoints of the cells
mid_x, mid_y = np.meshgrid(np.arange(cell_size/2, width, cell_size), np.arange(cell_size/2, height, cell_size))
# The vector in each cell, pointing from the midpoint of the cell to the middle of the exit.
# Used later in calculating the radial speed.
direction = np.zeros((mid_x.shape[0],mid_x.shape[0],2))
direction[:, :, 0] = mid_x - 20
direction[:, :, 1] = mid_y - 10
d_norm = np.sqrt(direction[:,:,0] * direction[:,:,0] + direction[:,:,1] * direction[:,:,1])

# We will calculate mean values of some part of the data recursively by taking a "chunk" of the data.
chunk = 1000 # chunk size

# The outer loop goes through the folders. The data from the simulations should be stored there in .npy.gz format.
mylist = ['taset0'] # name of the folder, where the data is; can be an array of folders
for i in range(0, len(mylist)):

    # The inner loop goes through the simulations (in this case it goes through just one simulation)
    for j in range(int(sys.argv[1]), int(sys.argv[1]) + 1):

	# Data of pedestrians in the room at different times (0="not in room", 1="in room").
        if os.path.exists("{}{}{}{}{}{}".format('simulation_data/', mylist[i], '/', 'in_room1', j, '.npy.gz')):
            in_room = np.loadtxt("{}{}{}{}{}{}".format('simulation_data/', mylist[i], '/', 'in_room1', j, '.npy.gz'))

        # Calculate number of pedestrians in room at different times
        sum_in_room = np.sum(in_room, axis=1)

        # The time steps when there are 190 pedestrians in the room
        time_interval1_start = np.where(sum_in_room == interval1_start)

        # Take the first instant when there are 190 pedestrians in the room.
        #
        # If there are no time steps when there are 190 pedestrians in the room (because two pedestrians have
        # evacuated simultaneously, and thus the number of pedestrians go from 191 to 189), take the times when
        # there are 189 pedestrians in the room.
        if np.size(time_interval1_start) == 0:
            time_interval1_start = np.where(sum_in_room == (interval1_start - 1))[0][0]
        else:
            time_interval1_start = np.where(sum_in_room == interval1_start)[0][0]

        # The time steps when there are 145 pedestrians in the room
        time_interval2_start = np.where(sum_in_room == interval2_start)

        # Take the first instant when there are 145 pedestrians in the room.
        #
        # If there are no time steps when there are 145 pedestrians in the room (because two pedestrians have
        # evacuated simultaneously and the number of pedestrians go from 146 to 144), take the times when
        # there are 144 pedestrians in the room.
        if np.size(time_interval2_start) == 0:
            time_interval2_start = np.where(sum_in_room == (interval2_start - 1))[0][0]
        else:
            time_interval2_start = np.where(sum_in_room == interval2_start)[0][0]

        # The time steps when there are 100 pedestrians in the room
        time_interval3_start = np.where(sum_in_room == interval3_start)

        # Take the first instant when there are 100 pedestrians in the room.
        #
        # If there are no time steps when there are 100 pedestrians in the room (because two pedestrians have
        # evacuated simultaneously and the number of pedestrians go from 101 to 99), take the times when
        # there are 99 pedestrians in the room.
        if np.size(time_interval3_start) == 0:
            time_interval3_start = np.where(sum_in_room == (interval3_start - 1))[0][0]
        else:
            time_interval3_start = np.where(sum_in_room == interval3_start)[0][0]

        # The time steps when there are 55 pedestrians in the room
        time_interval4_start = np.where(sum_in_room == interval4_start)

        # Take the first instant when there are 55 pedestrians in the room.
        #
        # If there is no time steps when there are 55 pedestrians in the room (because two pedestrians have
        # evacuated simultaneously and the number of pedestrians go from 56 to 54), take the times when
        # there are 54 pedestrians in the room.
        if np.size(time_interval4_start) == 0:
            time_interval4_start = np.where(sum_in_room == (interval4_start - 1))[0][0]
        else:
            time_interval4_start = np.where(sum_in_room == interval4_start)[0][0]

        # The time steps when there 10 pedestrians in the room
        time_interval4_end = np.where(sum_in_room == interval4_end)

        # Take the first instant when there are 10 pedestrians in the room.
        #
        # If there are no time steps when there are 10 pedestrians in the room (because two pedestrians have
        # evacuated simultaneously and the number of pedestrians go from 11 to 9), take the times when
        # there are 9 pedestrians in the room.
        if np.size(time_interval4_end) == 0:
            time_interval4_end = np.where(sum_in_room == (interval4_end - 1))[0][0]
        else:
            time_interval4_end = np.where(sum_in_room == interval4_end)[0][0]

	# Data of x-positions of pedestrians at different times.
        # NOTE! The data is sampled at a finer resolution, thus we take only every second element of the array.
        if os.path.exists("{}{}{}{}{}{}".format('simulation_data/', mylist[i], '/', 'positions_x', j, '.npy.gz')):
            positions_x = np.loadtxt("{}{}{}{}{}{}".format('simulation_data/', mylist[i], '/', 'positions_x', j, '.npy.gz'))
            positions_x = positions_x[0::2] # take every second element

        # Data of y-positions of pedestrians at different times.
        # NOTE! The data is sampled at a finer resolution, thus we take only every second element of the array.
        if os.path.exists("{}{}{}{}{}{}".format('simulation_data/', mylist[i], '/', 'positions_y', j, '.npy.gz')):
            positions_y = np.loadtxt("{}{}{}{}{}{}".format('simulation_data/', mylist[i], '/', 'positions_y', j, '.npy.gz'))
            positions_y = positions_y[0::2] # take every second element

        # Data of pedestrians' velocities x-component at different times.
        if os.path.exists("{}{}{}{}{}{}".format('simulation_data/', mylist[i], '/', 'velocities_x', j, '.npy.gz')):
            velocities_x = np.loadtxt("{}{}{}{}{}{}".format('simulation_data/', mylist[i], '/', 'velocities_x', j, '.npy.gz'))

        # Data of pedestrians' velocities y-component at different times.
        if os.path.exists("{}{}{}{}{}{}".format('simulation_data/', mylist[i], '/', 'velocities_y', j, '.npy.gz')):
            velocities_y = np.loadtxt("{}{}{}{}{}{}".format('simulation_data/', mylist[i], '/', 'velocities_y', j, '.npy.gz'))

        # Arrays to save the micro-macro converted data
        velocity_x = np.zeros((time_interval4_end - time_interval1_start, n, m), dtype=np.float16) # velocity x-component
        velocity_y = np.zeros((time_interval4_end - time_interval1_start, n, m), dtype=np.float16) # velocity y-component
        speed = np.zeros((time_interval4_end - time_interval1_start, n, m), dtype=np.float16) # speed
        density = np.zeros((time_interval4_end - time_interval1_start, n, m), dtype=np.float16) # density
        projection = np.zeros((time_interval4_end - time_interval1_start, n, m), dtype=np.float16) # radial speed

	# Loop through the data when the number of pedestrians in the room goes from 190 to 10.
        # Using the Voronoi-method derive the macroscopic quantities.
        for t in range(time_interval1_start, time_interval4_end):

            # Positions of pedestrians inside the room
            agents_in_room = np.where(in_room[t, :] == 1)[0] # which pedestrians are in the room
            n_agents_in_room = len(agents_in_room) # number of pedestrians in the room
            points = np.concatenate((np.reshape(positions_x[t, agents_in_room], (n_agents_in_room, 1)),
                                     np.reshape(positions_y[t, agents_in_room], (n_agents_in_room, 1))), axis=1)

            # x- and y-components of velocities of pedestrians in room
            x_component = velocities_x[t, agents_in_room]
            y_component = velocities_y[t, agents_in_room]

            # Create a Voronoi tessalation from pedestrian center points
            vor = Voronoi(points)

            # Add also the Voronoi regions on the rim to the tessalation
            #
            # new_vertices contains all the vertices in the tessalation
            # new_regions contains the vertices used for each Voronoi area
            #
            # https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram
            # https://gist.github.com/pv/8036995
            new_regions, new_vertices = voronoi_finite_polygons_2d(vor)

            # Loop through the Voronoi tessalations and calculate the density for each cell in the grid 
            # (Steffen B, Seyfried A (2010) Methods for measuring pedestrian density, flow, speed and direction
            # with minimal scatter. Physica A: Statistical mechanics and its applications 389(9):1902-1910)
            for r in range(0, len(new_regions)):
                region = new_regions[r]
                # Shapely Polygon object from Voronoi cell
                voronoi_cell = Polygon(shell=new_vertices[region]) & boundbox

                # Area of the Voronoi cell
                vor_area = voronoi_cell.area

                # Calculate minimal and maximal x- and y-coordinate values of the Voronoi cell
                minx, miny, maxx, maxy = voronoi_cell.bounds
                # Round the minimal and maximal values to belong to a cell in the square grid
                minx, miny, maxx, maxy = np.round(
                    (minx / cell_size, miny / cell_size, maxx / cell_size, maxy / cell_size)).astype(int)

		# Make sure that min and max values don't get out of bounds.
                minx = np.maximum(0, minx - 1)
                miny = np.maximum(0, miny - 1)
                maxx = np.minimum(m, maxx + 1)
                maxy = np.minimum(n, maxy + 1)

                # Loop over cells in the grid intersecting with the Voronoi cell.
                for x in range(minx, maxx):
                    for y in range(miny, maxy):
                        intersect_area = grids[x * n + y].intersection(voronoi_cell).area # intersecting area
                        # Calculate the contribution of the pedestrian to the density and velocity in the grid cell.
                        density[t - time_interval1_start, y, x] += intersect_area / vor_area
                        velocity_x[t - time_interval1_start, y, x] += intersect_area * x_component[r]
                        velocity_y[t - time_interval1_start, y, x] += intersect_area * y_component[r]

            # Finalize calculating the weighted density and velocity in the cell, by dividing it by the cell area
            density[t - time_interval1_start, :, :] /= cell_size * cell_size
            velocity_x[t - time_interval1_start, :, :] /= cell_size * cell_size
            velocity_y[t - time_interval1_start, :, :] /= cell_size * cell_size

            # Flip the density matrix upside down because of peculiar indexing in python
            density[t - time_interval1_start, :, :] = np.flipud(density[t - time_interval1_start, :, :])
            velocity_x[t - time_interval1_start, :, :] = np.flipud(velocity_x[t - time_interval1_start, :, :])
            velocity_y[t - time_interval1_start, :, :] = np.flipud(velocity_y[t - time_interval1_start, :, :])

            # Calculate speed in cells from the resultant velocity vectors
            speed[t - time_interval1_start, :, :] = np.hypot(velocity_x[t - time_interval1_start, :, :],
                                                            velocity_y[t - time_interval1_start, :, :])

            # Radial speed (calculate projections of actualized velocities on desired velocities)
            projection[t - time_interval1_start, :, :] = (velocity_x[t - time_interval1_start, :, :] *
                                                          direction[:, :, 0] + velocity_y[t -
                                                                                          time_interval1_start, :, :] *
                                                          direction[:, :, 1]) / d_norm

    # Save the length of the time intervals
    intervals = np.array((time_interval2_start - time_interval1_start, time_interval3_start - time_interval2_start,
                          time_interval4_start - time_interval3_start, time_interval4_end - time_interval4_start))
    np.save("{}{}{}{}{}{}".format('fields/', mylist[i], '/', 'intervals', j, '.npy'), intervals)

    # Save the macroscopic data of speed, density and radial speed in .hdf5 format for each time interval
    # NOTE: The data is not averaged over time. The averaging is done in "average_fields.py". If one wants
    # to save space the averaging should be performed already in this code.

    # First interval (190...145 agents in the room)
    with h5py.File("{}{}{}{}{}{}".format('fields/', mylist[i], '/', 'speed_interval1', j, '.hdf5'), 'w') as hf1:
        hf1.create_dataset("{}{}{}{}{}{}".format('fields/', mylist[i], '/', 'speed_interval1', j, '.npy.gz'),
                           data=speed[time_interval1_start - time_interval1_start:
                                      time_interval2_start - time_interval1_start, :, :])
    with h5py.File("{}{}{}{}{}{}".format('fields/', mylist[i], '/', 'density_interval1', j, '.hdf5')) as hf2:
        hf2.create_dataset("{}{}{}{}{}{}".format('fields/', mylist[i], '/', 'density_interval1', j, '.npy.gz'),
                           data=density[time_interval1_start - time_interval1_start:
                                        time_interval2_start - time_interval1_start, :, :])
    with h5py.File("{}{}{}{}{}{}".format('fields/', mylist[i], '/', 'projection_interval1', j, '.hdf5')) as hf3:
        hf3.create_dataset("{}{}{}{}{}{}".format('fields/', mylist[i], '/', 'projection_interval1', j, '.npy.gz'),
                           data=projection[time_interval1_start - time_interval1_start:
                                           time_interval2_start - time_interval1_start, :, :])

    # Second interval (145...100 agents in the room)
    with h5py.File("{}{}{}{}{}{}".format('fields/', mylist[i], '/', 'speed_interval2', j, '.hdf5'), 'w') as hf4:
        hf4.create_dataset("{}{}{}{}{}{}".format('fields/', mylist[i], '/', 'speed_interval2', j, '.npy.gz'),
                           data=speed[time_interval2_start - time_interval1_start:
                                      time_interval3_start - time_interval1_start, :, :])
    with h5py.File("{}{}{}{}{}{}".format('fields/', mylist[i], '/', 'density_interval2', j, '.hdf5')) as hf5:
        hf5.create_dataset("{}{}{}{}{}{}".format('fields/', mylist[i], '/', 'density_interval2', j, '.npy.gz'),
                           data=density[time_interval2_start - time_interval1_start:
                                        time_interval3_start - time_interval1_start, :, :])
    with h5py.File("{}{}{}{}{}{}".format('fields/', mylist[i], '/', 'projection_interval2', j, '.hdf5')) as hf6:
        hf6.create_dataset("{}{}{}{}{}{}".format('fields/', mylist[i], '/', 'projection_interval2', j, '.npy.gz'),
                           data=projection[time_interval2_start - time_interval1_start:
                                           time_interval3_start - time_interval1_start, :, :])


    # First interval (100...55 agents in the room)
    with h5py.File("{}{}{}{}{}{}".format('fields/', mylist[i], '/', 'speed_interval3', j, '.hdf5'), 'w') as hf7:
        hf7.create_dataset("{}{}{}{}{}{}".format('fields/', mylist[i], '/', 'speed_interval3', j, '.npy.gz'),
                           data=speed[time_interval3_start - time_interval1_start:
                                      time_interval4_start - time_interval1_start, :, :])
    with h5py.File("{}{}{}{}{}{}".format('fields/', mylist[i], '/', 'density_interval3', j, '.hdf5')) as hf8:
        hf8.create_dataset("{}{}{}{}{}{}".format('fields/', mylist[i], '/', 'density_interval3', j, '.npy.gz'),
                           data=density[time_interval3_start - time_interval1_start:
                                        time_interval4_start - time_interval1_start, :, :])
    with h5py.File("{}{}{}{}{}{}".format('fields/', mylist[i], '/', 'projection_interval3', j, '.hdf5')) as hf9:
        hf9.create_dataset("{}{}{}{}{}{}".format('fields/', mylist[i], '/', 'projection_interval3', j, '.npy.gz'),
                           data=projection[time_interval3_start - time_interval1_start:
                                           time_interval4_start - time_interval1_start, :, :])

    # First interval (190...145 agents in the room)
    with h5py.File("{}{}{}{}{}{}".format('fields/', mylist[i], '/', 'speed_interval4', j, '.hdf5'), 'w') as hf10:
        hf10.create_dataset("{}{}{}{}{}{}".format('fields/', mylist[i], '/', 'speed_interval4', j, '.npy.gz'),
                           data=speed[time_interval4_start - time_interval1_start:
                                      time_interval4_end - time_interval1_start, :, :])
    with h5py.File("{}{}{}{}{}{}".format('fields/', mylist[i], '/', 'density_interval4', j, '.hdf5')) as hf11:
        hf11.create_dataset("{}{}{}{}{}{}".format('fields/', mylist[i], '/', 'density_interval4', j, '.npy.gz'),
                           data=density[time_interval4_start - time_interval1_start:
                                        time_interval4_end - time_interval1_start, :, :])
    with h5py.File("{}{}{}{}{}{}".format('fields/', mylist[i], '/', 'projection_interval4', j, '.hdf5')) as hf12:
        hf12.create_dataset("{}{}{}{}{}{}".format('fields/', mylist[i], '/', 'projection_interval4', j, '.npy.gz'),
                           data=projection[time_interval4_start - time_interval1_start:
                                           time_interval4_end - time_interval1_start, :, :])



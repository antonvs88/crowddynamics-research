import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import sys

# Gather the essential simulation data from .hdf5 files and save them in .npy.gz format (compressed numpy data).
# The integration step used in the simulations is 0.001s. To save memory, sample data every 0.1s. x- and y-position data is sampled with a finer interval with a 0.05s step.
sample_step = 0.1
integration_step = 0.001
sample_interval = int(sample_step/integration_step)
sample_interval_finer = int((sample_step*0.5)/integration_step)

# The outer loop goes through the folders. The data from the .hdf5 files is extracted to these folders as .npy.gz files
mylist = ['taset500'] # name of the folder, where the data is; can be an array of names
for i in mylist:

    # The inner loop goes through the simulations (in this case it goes through just one simulation)
    for j in range(int(sys.argv[1]), int(sys.argv[1]) + 1):

        # Open the .hdf5 file for given simulation
        filename = "{}{}{}{}{}".format(i, '/', 'RoomEvacuationGame', j, '.hdf5')
        file = h5py.File(filename, 'r')

        # Fetch the group name
        def find_foo(name):
            if isinstance(file[name], h5py.Group):
                return name
        b=file.visit(find_foo)

        group = file[b]
        subgroup = group['roomevacuationgame']
        time_tot = subgroup['time_tot']
        in_goal = subgroup['in_goal']

        subgroup = group['agent']
        in_room1 = subgroup['in_room1']
        positions = subgroup['position']
        velocities = subgroup['velocity']

        subgroup = group['egressgame'] # comment if game is not played
        strategy = subgroup['strategy'] # comment if game is not played
        
        np.savetxt("{}{}{}{}{}".format(i, '/', 'time_tot', j, '.npy.gz'), time_tot[0::sample_interval])
        np.savetxt("{}{}{}{}{}".format(i, '/', 'in_goal', j, '.npy.gz'), in_goal[0::sample_interval])
        np.savetxt("{}{}{}{}{}".format(i, '/', 'in_room1', j, '.npy.gz'), in_room1[0::sample_interval])
        np.savetxt("{}{}{}{}{}".format(i, '/', 'positions_x', j, '.npy.gz'), positions[0::sample_interval_finer, :, 0])
        np.savetxt("{}{}{}{}{}".format(i, '/', 'positions_y', j, '.npy.gz'), positions[0::sample_interval_finer, :, 1])
        np.savetxt("{}{}{}{}{}".format(i, '/', 'velocities_x', j, '.npy.gz'), velocities[0::sample_interval, :, 0])
        np.savetxt("{}{}{}{}{}".format(i, '/', 'velocities_y', j, '.npy.gz'), velocities[0::sample_interval, :, 1])
        np.savetxt("{}{}{}{}{}".format(i, '/', 'strategy', j, '.npy.gz'), strategy[0::sample_interval]) # comment if game not played

    # Close the HDF5-file
    file.close()

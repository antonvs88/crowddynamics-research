import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import matplotlib
import matplotlib.patches as ptch
from matplotlib.artist import setp
from matplotlib.collections import PatchCollection
from shapely.geometry import Point, Polygon, LineString
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.gridspec as gridspec
import matplotlib.ticker
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter

# Folders containing the field data averaged over time and sample for different scenarios (npy.gz format).
# NOTE: Check that the location is correct!
mylist = ['taset0', 'taset80', 'taset125', 'taset150', 'taset220', 'taset350', 'taset500', 'taset1000', 'vd1_A2000', 'overtaking']

# Figure titles
titles = [r'$T_{ASET}=0$', r'$T_{ASET}=80$', r'$T_{ASET}=125$', r'$T_{ASET}=150$', r'$T_{ASET}=220', r'$T_{ASET}=350$', r'$T_{ASET}=500$', r'$T_{ASET}=1000$', 'all impatient']

faces = ['red', 'cyan', 'none', 'magenta', 'none', 'none', 'blue'] # scatter plot marker face color (elements 0,1,3,6 are used)
edges = ['darkred', 'c', 'none', 'm', 'none', 'none', 'darkblue'] # scatter plot marker edge color (elements 0,1,3,6 are used)
markers = ["^", "v", ".", "s", ".",".", "^"] # scatter plot marker types (elements 0,1,3,6 are used)

# Number of agents in room initially
n_a = 200

# Simulations used in the plots
run_start = 0
run_end = 100
run_len = run_end - run_start

# An arbitrary upper limit given to the number of seconds it takes for the crowd to evacuate
arbitrary_end = 600
# Time step (this is the maximum resolution, if one wants to use a better resolution this has to be done in "hdf5_to_npy_gz.py")
interval = 0.1
time_len = int(arbitrary_end / interval) # number of time steps

# Arrays to store data used in plots
time_steps_lst = np.arange(0, arbitrary_end, interval) # array for time steps
end_times_lst = np.zeros((len(mylist) - 1, run_len)) # array for end time for each scenario and simulation
strat_lst = np.zeros((len(mylist) - 2, run_len)) # array for proportion of imp. peds. when 10 pedestrians have evacuated
impatients = 0 * np.ones((run_len, time_len)) # array for number of impatient pedestrians in the room (for scenario with fixed strategies)
patients = 0 * np.ones((run_len, time_len)) # array for number of patient pedestrians in the room (for scenario with fixed strategies)
avg_init_prop = np.zeros(len(mylist) - 1) # average initial proportion of impatient agents (for scenarios with game)
avg_flow = np.zeros(len(mylist) - 1) # average flow over whole evacuation (for scenarios with game)
sem = np.zeros(len(mylist) - 1) # standard mean error of flow for whole evacuation (for scenarios with game)
std = np.zeros(len(mylist) - 1) # standar deviation of flow for whole evacuation (for scenarios with game)

# Initialize the figure
grid_cell = 10.5
y_disp = 1
x_disp = 1
fig = plt.figure(0)
G = gridspec.GridSpec(int(2*grid_cell), int(2*grid_cell))

# Subplot for the empirical time lapse survival function (for scenarios with game, Taset=500, 150, 80, 0)
ax0 = plt.subplot(G[int(np.ceil(grid_cell)) + 2*y_disp:int(2*grid_cell), int(np.floor(grid_cell/2)):int(np.floor(3/2*grid_cell))])

# Subplot for overtaking figure (for scenario with fixed strategies)
ax1 = plt.subplot(G[0:int(np.floor(grid_cell))-y_disp, 0:int(np.floor(grid_cell))-x_disp])

# Subplot for the Taset vs average flow figure (for scenarios with game)
ax2 = plt.subplot(G[0:int(np.floor(grid_cell))-y_disp, int(np.ceil(grid_cell))+x_disp:int(2*grid_cell)])

# Loop through all the scenarios, collect data, and plot data from scenarios to the subplots.
for i in range(0, len(mylist)):

    # Time lapses array contains the points in time when a pedestrian has exited the room.
    time_lapses = np.zeros((run_len * n_a))
    t = 0 # help variable for time lapses vector

    # Loop through all the simulations
    for j in range(run_start, run_end): 

	# Open data for simulation time points
        if os.path.exists("{}{}{}{}{}{}".format('simulation_data/', mylist[i], '/', 'time_tot', j, '.npy.gz')):
            time_tot = np.loadtxt("{}{}{}{}{}{}".format('simulation_data/', mylist[i], '/', 'time_tot', j, '.npy.gz'))
            # For simulations with the game, save the end time of the simulation.
            if i < 9:
                end_times_lst[i,j] = time_tot[-1]

        # Data of pedestrians in the room at different times (0="not in room", 1="in room").
        if os.path.exists("{}{}{}{}{}{}".format('simulation_data/', mylist[i], '/', 'in_room1', j, '.npy.gz')):
            in_room1 = np.loadtxt("{}{}{}{}{}{}".format('simulation_data/', mylist[i], '/', 'in_room1', j, '.npy.gz'))

        # For simulations with the game, open data for number of pedestrians that have left the room.
        if i < 9:
            if os.path.exists("{}{}{}{}{}{}".format('simulation_data/', mylist[i], '/', 'in_goal', j, '.npy.gz')):
                in_goal = np.loadtxt("{}{}{}{}{}{}".format('simulation_data/', mylist[i], '/', 'in_goal', j, '.npy.gz'))

            # Difference the time series of number of evacuated pedestrians
            diff_in_goal = np.ediff1d(in_goal)
        
            # In the differenced time series array, find elements equaling 1 or 2.
            # These correspond to instances when 1 or 2 pedestrians have evacuated.
            # The door is not wide enough for 3 pedestrians to evacuate simultaneously.
            indx_in_goal = np.where((1 <= diff_in_goal) & (diff_in_goal <= 2))
            in_goal_times = time_tot[indx_in_goal]
        
            # Find the times when two pedestrians evacuate simultaneously.
            # Duplicate these times in the "in_goal_times" array.
            if np.any(diff_in_goal == 2):
                two_exited_indices = np.where(diff_in_goal[indx_in_goal] == 2)
                two_exited_times = in_goal_times[two_exited_indices]
                in_goal_times = np.insert(in_goal_times, two_exited_indices[0], two_exited_times)
        
            # Add a zero to the beginning of the "in_goal_times" array to aid in calculations.
            in_goal_times = np.insert(in_goal_times, 0, 0)
        
            # Time lapses between evacuated pedestrians
            time_lapses[t:t+len(in_goal_times)-1] = np.ediff1d(in_goal_times)
            t += len(in_goal_times) - 1

        # Data of strategies of pedestrians in the room at different times (0="impatient", 1="patient")
        if os.path.exists("{}{}{}{}{}{}".format('simulation_data/', mylist[i], '/', 'strategy', j, '.npy.gz')):
            strategy = np.loadtxt("{}{}{}{}{}{}".format('simulation_data/', mylist[i], '/', 'strategy', j, '.npy.gz'))

        # Calculate the proportion of impatient pedestrians in the room after 10 pedestrians have evacuated,
        # i.e., approximately at the time when the crowd has formed a half-circle in front of the exit.
        if i < 8:
            # The time step when the 10th pedestrian has evacuated
            ind_stat_start = np.where(in_goal == 10)
        
            # Take into account that the 10th and 11th pedestrians might evacuate simultaneously
            if len(ind_stat_start[0]) == 0:
                ind_stat_start = np.where(in_goal == 11)

            if len(ind_stat_start[0]) > 1:
                ind_stat_start = ind_stat_start[0][0]
            else:
                ind_stat_start = ind_stat_start[0][0]

            # Which pedestrians are in the room, when 10 pedestrians have exited.
            start_in_room = np.where(in_room1[ind_stat_start, :] == 1)[0]

            # Proportion of impatient pedestrians in the room, when 10 pedestrians have exited.
            #print(strategy[ind_stat_start, start_in_room])
            strat_lst[i,j] = 1 - np.sum(strategy[ind_stat_start, start_in_room]) / np.sum(in_room1[ind_stat_start, :])

        # For the simulation with fixed strategies calculate the proportion of impatient pedestrians
        # in the room at every time step.
        if i == 9:

            iterations = time_tot.shape[0] # number of time steps in the simulations alltogether
            impatient_inroom = np.zeros(iterations) # number of impatient pedestrians in room at different times
            patient_inroom = np.zeros(iterations) # number of patient pedestrians in room at different times

            for k in range(0, iterations):
                not_exited = np.where(in_room1[k, :] == 1)[0] # indices of pedestrians in room at iteration k
                n_inroom = len(not_exited) # number of pedestrians in room at iteration k
                strats = strategy[k, not_exited] # strategies of pedestrians in room at iteration k
                patient_inroom[k] = np.sum(strats) # number of patient pedestrians in room at iteration k
                impatient_inroom[k] = n_inroom - patient_inroom[k] # number of impatient pedestrians in room at iteration k

            impatients[j,0:iterations] = impatient_inroom # number of impatient pedestrians at different times in simulation j
            patients[j,0:iterations] = patient_inroom # number of patient pedestrians at different times in simulation j

    # Plot time lapse survival functions for Taset=0, 80, 150, 500
    # time lapse resolution 0.1, because sample step 0.1
    if i in {0,1,3,6}:
        # Modify time lapses data so that np.unique can be used on them
        #print(titles[i]) # print scenario name
        #print(np.mean(time_lapses[0:t])) # print mean of time lapses
        #print(np.std(time_lapses[0:t])) # print standard error of time lapses
        #print(np.std(time_lapses[0:t])/np.sqrt(len(time_lapses[0:t]))) # print standard error of mean
        time_lapses = np.array(time_lapses[0:t])*10 # time lapses multiplied by 10 (get to int)
        time_lapses = time_lapses.astype(int) # change type to int
        time_lapses = np.sort(time_lapses) # sort datapoints
        n_datapoints = len(time_lapses) # number of datapoints
        
        # Create an array of unique data points
        unique_points, up_indices, up_counts = np.unique(time_lapses, return_index=True, return_counts=True)
        up_indices = up_indices + up_counts - 1

        # The cumulative distribution function can be calculated, when we know the number of data points
        # n. Denote a data point with x_i, i=0,...,n, where the data points are ordered from smallest to largest
        # value. The cdf for a data point x_i is then i/n.
        ordering = np.arange(0, n_datapoints) # order number of datapoints
        cdf = ordering/n_datapoints # empiric cumulative distribution function P(dt < t)
        ccdf = 1 - cdf # empiric survival function P(dt >= t)

        if i == 0:
            ax0.scatter(unique_points/10, ccdf[up_indices], marker=markers[i], s = 40,
                        facecolors=faces[i], edgecolors=edges[i], label=titles[i], linewidth=0.1)
        if i == 1:
            ax0.scatter(unique_points/10, ccdf[up_indices], marker=markers[i], s = 30,
                        facecolors=faces[i], edgecolors=edges[i], label=titles[i], linewidth=0.1)
        if i == 3:
            ax0.scatter(unique_points/10, ccdf[up_indices], marker=markers[i], s = 30,
                        facecolors=faces[i], edgecolors=edges[i], label=titles[i], linewidth=0.1)
        if i == 6:
            ax0.scatter(unique_points/10, ccdf[up_indices], marker=markers[i], s = 40,
                        facecolors=faces[i], edgecolors=edges[i], label=titles[i], linewidth=0.1)
        ax0.set_yscale('log')
        ax0.set_xscale('log')
        ax0.set_xlabel('Time, t (s)', fontsize=12)
        ax0.set_ylabel(r'Probability of $\Delta$x > t', fontsize=12)
        ax0.get_xaxis().set_tick_params(direction='out', width=2, top='off')
        ax0.yaxis.set_major_formatter(ScalarFormatter())
        ax0.xaxis.set_major_formatter(ScalarFormatter())
        ax0.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax0.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax0.get_yaxis().set_tick_params(direction='out', width=2, top='off', labelsize=12, pad=0.05)
        ax0.get_xaxis().set_tick_params(direction='out', width=2, top='off', labelsize=12, pad=0.05)
        ax0.tick_params(axis='x', which='minor', direction='out', width=2)
        ax0.tick_params(axis='y', which='minor', direction='out', width=2)
        ax0.get_xaxis().tick_bottom()
        ax0.get_yaxis().tick_left()
        ax0.yaxis.set_label_coords(-0.21,0.5)
        ax0.set_ylim(0.00003,1)
        ax0.set_xlim(right=20)
        ax0.set_xlim(left=0.1)
        ax0.set_yscale('log')
        ax0.legend(loc='lower left',
                   fontsize=12,
                   scatterpoints=1,
                   bbox_to_anchor=(0, 0),
                   handletextpad=0.1,
                   borderpad=0.25,
                   borderaxespad=0.25,
                   )

    # Plot the overtaking figure, i.e., how many impatient and patient pedestrians there are in the room at different
    # times (for scenario with fixed strategies).
    if i == 9:
        ax1.plot(time_steps_lst, 100 - np.mean(impatients, axis=0), linewidth=2, color="black", label="Impatient",
                 linestyle='-')
        ax1.plot(time_steps_lst, 100 - np.mean(patients, axis=0), linewidth=2, color="black", label="Patient",
                 linestyle='-.')
        ax1.set_yticks([0,25,50,75,100])
        ax1.set_xticks([0,75,150,225])
        ax1.set_yticklabels([0,25,50,75,100], fontsize=12)
        ax1.set_xticklabels([0,75,150,225], fontsize=12)
        ax1.get_yaxis().set_tick_params(direction='out', width=2, top='off', labelsize=12)
        ax1.get_xaxis().set_tick_params(direction='out', width=2, top='off', labelsize=12)
        ax1.set_ylim(0,100)
        ax1.set_xlim(0,225)
        ax1.set_ylabel('Number of evacuated pedestrians', fontsize=12)
        ax1.set_xlabel(r'Time, t (s)', fontsize=12)
        ax1.legend(loc='lower right',
                   fontsize=12,
                   bbox_to_anchor=(1.0, 0.0),
                   handletextpad=0.1,
                   borderpad=0.25,
                   borderaxespad=0.25,
                   )

    # Plot the Taset vs average flow figure (for scenarios with game).
    if i < 9:
        if i < 8:
            avg_init_prop[i] = np.mean(strat_lst[i,:]) # average initial prop. of impatient pedestrians
            avg_flow[i] = 200/(1.2*np.mean(end_times_lst[i,:])) # average flow at exit
            ax2.scatter(avg_init_prop[i], avg_flow[i], facecolors='black', edgecolors='black')
        # "All patient pedestrians" scenario
        if i == 8:
            avg_init_prop[i] = 0 # average initial prop. of impatient pedestrians is obviously 0
            avg_flow[i] = 200/(1.2*np.mean(end_times_lst[i,:])) # average flow at exit
            ax2.scatter(avg_init_prop[i], avg_flow[i], facecolors='black', edgecolors='black')

        # Calculate standard error of flow mean. The values are so small that they don't show up on the plot!
        sem[i] = np.std(200/(1.2*end_times_lst[i,:])) / np.sqrt(run_len)

        # Calculate standar deviation of flow
        std[i] = np.std(200/(1.2*end_times_lst[i,:]))

        # Plot errorbar with standard deviation
        ax2.errorbar(avg_init_prop[i], avg_flow[i], yerr=std[i], ecolor='black', ls='none', capthick=1, capsize=4)

        ax2.set_yticks([0,0.4,0.8,1.2,1.6])
        ax2.set_xticks([0,0.25,0.5,0.75,1.0])
        ax2.set_yticklabels([0,0.4,0.8,1.2,1.6])
        ax2.set_xticklabels([0.0,0.25,0.50,0.75,1.0])
        ax2.get_yaxis().set_tick_params(direction='out', width=2, top='off', labelsize=12, pad=0.05)
        ax2.get_xaxis().set_tick_params(direction='out', width=2, top='off', labelsize=12, pad=0.05)
        ax2.set_xlim(0,1)
        ax2.set_ylim(0,1.6)
        ax2.set_ylabel(r'Pedestrian flow at exit (1/(m$\cdot$s)) ', fontsize=12)
        ax2.set_xlabel('Proportion of Impatient', fontsize=12)

# Plot some details on the Taset vs average flow figure
ax2_point1 = [0.07, 0.008]
ax2_point2 = [0.8, 0.96]
ax2.plot(ax2_point1, ax2_point2, color="black", linewidth=1.25)
ax2.text(0.08, 0.6, '(all pedestrians' + '\n' + 'patient)', fontsize=12)

# Plot figure letters
fig.text(0.003, 0.94, 'a', fontweight='bold', fontsize=16, transform=fig.transFigure)
fig.text(0.45, 0.94, 'b', fontweight='bold', fontsize=16, transform=fig.transFigure)
fig.text(0.2, 0.43, 'c', fontweight='bold', fontsize=16, transform=fig.transFigure)

# Make a second x-axis for the Taset vs average flow figure, with Taset values.
ax3 = ax2.twiny()
ax2.set_xlim(-0.05,1.05)
ax3.set_xlim(-0.05,1.05)
ax3.xaxis.set_ticks_position("top")
ax3.xaxis.set_label_position("top")
ax3.spines["top"].set_bounds(0.2,1)
ax3.set_frame_on(True)
ax3.patch.set_visible(False)
ax3.set_xlabel(r"$T_{ASET}$", fontsize=12)
ax3.set_xticks([0.25,0.5,0.75,1.0])
ax3.get_xaxis().set_tick_params(direction='out', width=2, pad=0.05)
ax3.set_xticklabels([500,150,80,0], fontsize=12)

plt.savefig('figure_3.pdf',
            bbox_inches='tight'
            )



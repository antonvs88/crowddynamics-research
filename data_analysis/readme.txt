Here are instructions for using the data analysis and figure plotting codes in the folder.
NOTE! The locations for the data should be changed in the code.

hdf5_to_npy_gz.py
Simulation codes stores data in .hdf5 files. This python file retrieves data from the .hdf5 files and stores it in .npy.gz files.

hdf5_to_npy_gz_script.sh
Shell script to run "hdf5_to_npy_gz.py" for multiple simulations. NOTE! The locations of the folders should be changed.

calculate_field_data.py
Take the microscopic data from .npy.gz files and convert it into macroscopic field data (density, speed, radial speed) with Voronoi
method.

field_data_script.sh
Shell script to run "calculate_field_data.py" for multiple simulations. NOTE! The locations of the folders should be changed.

voronoi_finite_polygons_2d.py
Reconstruct infinite voronoi regions in a 2D diagram to finite regions. "calculate_field_data.py" calls this function.

average_fields.py
Take field data created by "calculate_field_data.py" and average it to calculate average density, speed, radial speed, crowd
pressure and radial crowd pressure fields.

recursive_mean.py
Calculate mean of data array recursively by averaging a "chunk" of the data. "average_fields.py" calls this function.

figure_1.py
Create Figure 1 of von Schantz & Ehtamo "Overtaking others in a spatial game of egress congestion".

figure_2.py
Create Figure 2 of von Schantz & Ehtamo "Overtaking others in a spatial game of egress congestion".

figure_3.py
Create Figure 3 of von Schantz & Ehtamo "Overtaking others in a spatial game of egress congestion".

figure_4.py
Create Figure 4 of von Schantz & Ehtamo "Overtaking others in a spatial game of egress congestion".

figure_SI_temporal_evolution_fields.py
Create Figure 5 and 6 of Supplemental Information of von Schantz & Ehtamo "Overtaking others in a spatial game of egress
congestion".

radial_mean.py
An approximate method to calculate average speed and crowd pressure at different distances from the exit. Points at equal spacing
along a semicircle are generated, and then it is identified to which cells in the grid these points belong. The average speed and
crowd pressure at the distance in question is calculated by averaging over the data in the cells. "figure_4.py" calls this
function.

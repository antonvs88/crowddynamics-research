import numpy as np


def radial_mean(speed, crowd_pressure, cell_size, width, height):
    """
    An approximate method to calculate average speed and crowd pressure at different distances
    from the exit. Points at equal spacing along a semicircle are generated, and then it is
    identified to which cells in the grid these points belong. The average speed and crowd pressure
    at the distance in question is calculated by averaging over the data in the cells.

    Parameters
    ----------
    speed : array
    Array containing the average speed field.
    crowd_pressure : array
        Array containing the average crowd pressure field.
    cell_size:
        Size of the cell in the grid spanning the room.
    width: integer
        Width of the room.
    height:
        Length of the room.

    Returns
    -------
    distances: array
        Distances to exit.
    avg_speed : array
        Average speed at different distances to exit.
    avg_crowd_pressure : array
        Average crowd pressure at different distances to exit.

    """


    if width != height:
        raise ValueError("code designed for square rooms")

    width = width / cell_size
    width = int(width)
    distances = np.arange(0, width / 2, 1, dtype=np.int16)
    avg_speed = np.zeros(len(distances))  # array for storing average speed at different distances
    avg_crowd_pressure = np.zeros(len(distances))  # array for storing average crowd pressure
    # Loop through different distances
    for dist_indx in range(0, len(distances)):
        radius = distances[dist_indx]  # distance
        x = np.arange(width - radius, width + 1, 1)  # x-values in the range of half-circle
        x_flip = x[::-1]
        x = np.concatenate((x_flip, x[1:len(x)]), axis=0)  # y-values in the range of half-circle

        # y = y0 +- sqrt(r^2 - (x-x0)^2)
        periphery_y1 = width / 2 + np.sqrt(radius * radius - (x_flip - width) * (x_flip - width))
        periphery_y2 = width / 2 - np.sqrt(radius * radius - (x_flip - width) * (x_flip - width))
        periphery_y = np.concatenate((periphery_y1[:-1], periphery_y2[::-1]), axis=0)

        # Given a distance, calculate the average speed and crowd pressure.
        # Use data in the cells that intersect with the half circle.
        avg_speed[dist_indx] = np.mean(speed[(periphery_y - 1).astype(int), (x - 1).astype(int)])
        avg_crowd_pressure[dist_indx] = np.mean(crowd_pressure[(periphery_y - 1).astype(int), (x - 1).astype(int)])

    # Change distances array data back to meters
    distances = cell_size * distances

    return distances, avg_speed, avg_crowd_pressure

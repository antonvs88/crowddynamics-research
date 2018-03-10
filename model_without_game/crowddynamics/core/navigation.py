import logging
from collections import Iterable

import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Polygon
from shapely.geometry import MultiLineString
from shapely.geometry import mapping

from crowddynamics.core.geometry import shapes_to_point_pairs
from crowddynamics.core.vector2D import angle_nx2

try:
    import skfmm
    import skimage.draw
except ImportError:
    raise Warning("Navigation algorithm cannot be used if scikit-fmm or "
                  "scikit-image are not installed")

from .vector2D import angle_nx2


def plot_dmap(grid, dmap, phi, name):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    X, Y = grid
    plt.figure(figsize=(12, 12))
    plt.title('Distance map from exit.')
    plt.imshow(dmap, interpolation='bilinear', origin='lower', cmap=cm.gray,
               extent=(X.min(), X.max(), Y.min(), Y.max()))
    plt.contour(X, Y, dmap, 30, linewidths=1, colors='gray')
    plt.contour(X, Y, phi.mask, [0], linewidths=1, colors='black')
    plt.savefig("distance_map_{}.pdf".format(name))


class ExitSelection:
    """Exit selection policy."""

    def __init__(self, simulation):
        self.simulation = simulation


class NavigationMap(object):
    initial = -1.0
    target = 1.0
    obstacle = True

    def __init__(self, domain, step=0.01): # NOTE: changing step parameter has influence on navigation in bottleneck
        self.domain = domain
        self.step = step

        minx, miny, maxx, maxy = domain.bounds  # Bounding box
        print(domain.bounds)
        self.grid = np.meshgrid(np.arange(minx, maxx + step, step=step),
                                np.arange(miny, maxy + step, step=step), )

    def points_to_indices(self, points):
        return np.round(points / self.step).astype(int)

    def set_values(self, shape, array, value):
        if isinstance(shape, Polygon):
            points = np.asarray(shape.exterior)
            points = self.points_to_indices(points)
            x, y = points[:, 0], points[:, 1]
            j, i = skimage.draw.polygon(x, y)
            array[i, j] = value
        elif isinstance(shape, LineString):
            points = shapes_to_point_pairs(shape)
            points = self.points_to_indices(points)
            for args in points:
                j, i = skimage.draw.line(*args.flatten())
                array[i, j] = value
        elif isinstance(shape, Iterable):
            for shape_ in shape:
                self.set_values(shape_, array, value)
        else:
            raise Exception()

    def distance_map(self, obstacles, targets):
        np.set_printoptions(threshold=np.nan)

        contour = np.full_like(self.grid[0], self.initial, dtype=np.float64)
        self.set_values(targets, contour, self.target)

        mask = np.full_like(self.grid[0], False, dtype=np.bool_)
        self.set_values(obstacles, mask, self.obstacle)
        contour = np.ma.MaskedArray(contour, mask)
        dmap = skfmm.distance(contour, dx=self.step)
        return dmap, contour

    def travel_time_map(self):
        pass

    def static(self):
        pass

    def dynamic(self, obstacles, targets, dynamic):
        pass


class Navigation(NavigationMap):
    """Determining target direction of an agent in multi-agent simulation.

    Algorithm based on solving the continous shortest path
    problem by solving eikonal equation. [1]_, [2]_

    There are at least two open source eikonal solvers. Fast marching method
    (FMM) [3]_ for rectangular and tetrahedral meshes using Python and C++ and
    fast iterative method (FIM) [4]_ for triangular meshes using c++ and CUDA.

    In this implementation we use the FMM algorithm because it is simpler.

    .. [1] Kretz, T., Große, A., Hengst, S., Kautzsch, L., Pohlmann, A., & Vortisch, P. (2011). Quickest Paths in Simulations of Pedestrians. Advances in Complex Systems, 14(5), 733–759. http://doi.org/10.1142/S0219525911003281
    .. [2] Cristiani, E., & Peri, D. (2015). Handling obstacles in pedestrian simulations: Models and optimization. Retrieved from http://arxiv.org/abs/1512.08528
    .. [3] https://github.com/scikit-fmm/scikit-fmm
    .. [4] https://github.com/SCIInstitute/SCI-Solver_Eikonal
    """

    # TODO: take into account radius of the agents

    def __init__(self, simulation):
        super().__init__(simulation.domain)

        self.simulation = simulation
        self.dist_map1 = None
        self.direction_map1 = None
        self.dist_map2 = None
        self.direction_map2 = None

    def static_potential(self):
        logging.info("")

        r = 0
        height = self.simulation.height
        width = self.simulation.width
        door_width = self.simulation.door_width

        # Instead of one distance map, there are two distance maps, one for each room.

        walls1 = MultiLineString([((r,r), (r,height-r), (width-r,height-r), (width-r,(height+door_width)/2-r), \
                         (width+r, (height+door_width)/2-r), (width+r, height-r), (2*width-r, height-r), \
                         (2*width-r,r), (width+r,r), (width+r, (height-door_width)/2+r), \
                         (width-r, (height-door_width)/2+r), (width-r,r), (r,r))])

        walls2 = MultiLineString([((r, r), (r, height - r), (2 * width - r, height - r), (2 * width - r, r), (r, r))])

        new_obstacles1 = []
        new_obstacles1.append(walls1)

        new_obstacles2 = []
        new_obstacles2.append(walls2)

        new_goal = LineString([(2*width-2*0.3, height - 2*0.3), (2*width-2*0.3, 2*0.3)])
        new_exits = []
        new_exits.append(new_goal)

        if self.dist_map1 is None:
            self.dist_map1, contour1 = self.distance_map(
                new_obstacles1,
                new_exits,
            )

        if self.dist_map2 is None:
            self.dist_map2, contour2 = self.distance_map(
                new_obstacles2,
                new_exits,
            )

        # Plot the distance maps
        #plot_dmap(self.grid, self.dist_map1, contour1, 'map1')
        #plot_dmap(self.grid, self.dist_map2, contour2, 'map2')

        # Save the distance map data
        #np.savetxt('dmap1.out', self.dist_map1)
        #np.savetxt('dmap2.out', self.dist_map1)

        u, v = np.gradient(self.dist_map1)
        l = np.hypot(u, v)  # Normalize
        direction = np.zeros(u.shape + (2,))
        # Flip order from (row, col) to (x, y)
        direction[:, :, 0] = v / l
        direction[:, :, 1] = u / l
        self.direction_map1 = direction

        u, v = np.gradient(self.dist_map2)
        l = np.hypot(u, v)  # Normalize
        direction = np.zeros(u.shape + (2,))
        # Flip order from (row, col) to (x, y)
        direction[:, :, 0] = v / l
        direction[:, :, 1] = u / l
        self.direction_map2 = direction

    def distance_map_agents(self):
        pass

    def dynamic_potential(self):
        logging.info("")

        raise NotImplementedError

    def update(self):
        i = self.simulation.agent.indices()
        points = self.simulation.agent.position[i]
        indices = self.points_to_indices(points)
        indices = np.fliplr(indices)
        # http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
        # TODO: Handle index out of bounds -> numpy.take
        self.simulation.agent.target_direction[self.simulation.agent.in_room1] = self.direction_map1[indices[self.simulation.agent.in_room1, 0], indices[self.simulation.agent.in_room1, 1], :]
        self.simulation.agent.target_direction[~self.simulation.agent.in_room1] = self.direction_map2[indices[~self.simulation.agent.in_room1, 0], indices[~self.simulation.agent.in_room1, 1], :]

class Orientation:
    """
    Target orientation
    """

    def __init__(self, simulation):
        self.simulation = simulation

    def update(self):
        if self.simulation.agent.orientable:
            dir_to_orient = angle_nx2(self.simulation.agent.target_direction)
            self.simulation.agent.target_angle[:] = dir_to_orient

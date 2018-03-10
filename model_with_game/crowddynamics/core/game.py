import numba
import numpy as np
from matplotlib.path import Path
from numba.types import deferred_type
from shapely.geometry import Polygon
from scipy.stats import erlang

from crowddynamics.core.vector2D import length_nx2, length

@numba.jit(nopython=True)
def payoff(s_our, s_neighbor, t_aset, t_evac_i, t_evac_j):
    """Payout from game matrix.

    :param s_our: Our strategy
    :param s_neighbor: Neighbor strategy
    """
    average = (t_evac_i + t_evac_j) / 2
    if t_aset == 0:
        taset_ratio = np.inf
    else:
        taset_ratio = average / t_aset

    if s_neighbor == 0:
        if s_our == 0:
            if taset_ratio == 0:
                return np.inf
            elif taset_ratio == np.inf:
                return 0
            else:
                return 1 / taset_ratio
        elif s_our == 1:
            return 1
    elif s_neighbor == 1:
        if s_our == 0:
            return -1
        elif s_our == 1:
            return 0
#    else:
#        raise Exception("Not valid strategy.")


@numba.jit(nopython=True)
def agent_closer_to_exit(points, position):
    mid = (points[0] + points[1]) / 2.0
    dist = length_nx2(mid - position)
    num = np.argsort(dist)
    num = np.argsort(num)
    return num

@numba.jit(nopython=True)
def exit_capacity(points, agent_radius):
    """Capacity of narrow exit."""
    door_radius = length(points[1] - points[0]) / 2.0
    capacity = door_radius // agent_radius
    return capacity

@numba.jit(nopython=True)
def numba_cumsum(A):
    """"np.cumsum(A,axis=1) for numba"""
    maxR = len(A[:,0])
    maxC = len(A[0,:])
    cumsumC = np.zeros((maxR,maxC))
    for i in range(maxR):
        cumsumC[i][:] = np.cumsum(A[i][:])
    return cumsumC

#TODO: scipy.stats.erlang.sf doesn't work with Numba
@numba.jit(nopython=False)
def erlang_shape(interval, dt):
    """
    Calculate the maximum amount of times an agent can update its strategy
    during the time step dt, if the update times are exponentially distributed,
    and the parameter interval corresponds to the average time interval between
    two strategy updates.
    """
    sf = 0
    shape_param = 0
    while (sf < 1):
           shape_param = shape_param + 1
           sf = erlang.sf(dt, shape_param, loc=0, scale=interval)
    return shape_param

#@numba.jit(nopython=True)
def poisson_update(agent, players, door, radius_max, strategy,
                   strategies, t_evac, t_aset, interval, dt):
    """
    During the time step dt, each agent updates its strategy according to
    independent identical Poisson processes, with the parameter "interval"
    corresponding to the average time between two Poisson arrivals. 
    """
    x = agent.position[players] # retrieve the positions (x,y) of the agents
    """Calculate the estimated evacuation times of all the agents."""
    """ Exit capacity 1.25 (1/s) is used in previous research, but this can of course vary."""
    #t_evac = agent_closer_to_exit(door, x) / exit_capacity(door, radius_max)
    t_evac[players] = agent_closer_to_exit(door, x) / 1.25
    loss = np.zeros(2)  # values: loss, indices: strategy
    n_a = len(players) # number of agents playing the game
#    shape_param = erlang_shape(interval, dt) # calculate the maximum amount of times an agent can update its strategy, if erlang_shape works
    shape_param = 10
    """Generate k exponentially distributed update times for each agent."""
    # TODO: In order to replicate the results ,the seed number should be specified?
    upd_times = np.random.exponential(scale=interval, size=(n_a,shape_param))
    a = numba_cumsum(upd_times) # calculate the cumulative sum of update times for each agent
    """Choose all the elements which satisfy <= dt."""
    b = np.where(a <= dt)
    """ Numba doesn't allow advanced indexing for more than one instance. A trick to overcome this."""
    c = np.zeros(len(b[0]))
    for i in range(len(b[0])):
        c[i] = a[b[0][i], b[1][i]]
    "Sort the elements"
    d = np.argsort(c)
    e = players[b[0][d]]
    """Loop through the agents in correct chronological order,
    and set the agents to update to their best-response strategy.
    Strategies: {0: "Impatient", 1: "Patient"}."""
    for i in e:
        for j in agent.neighbors[i]:
            if j < 0:
                continue
            else:
                for s_our in strategies:
                    loss[s_our] += payoff(s_our, strategy[j], t_aset, t_evac[i], t_evac[j])
        strategy[i] = np.argmin(loss)  # Update strategy
        loss[:] = 0 # Reset loss array

class EgressGame(object):
    """
    Patient and impatient pedestrians in a spatial game for egress congestion
    -------------------------------------------------------------------------
    Strategies: {0: "Impatient", 1: "Patient"}.

    .. [1] Heli??vaara, S., Ehtamo, H., Helbing, D., & Korhonen, T. (2013). Patient and impatient pedestrians in a spatial game for egress congestion. Physical Review E - Statistical, Nonlinear, and Soft Matter Physics. http://doi.org/10.1103/PhysRevE.87.012802
    """

    def __init__(self, simulation, door, room1, t_aset_0,
                 interval=0.001, neighbor_radius=0.6, neighborhood_size=100):
        """
        Parameters
        ----------
        :param simulation: MultiAgent Simulation
        :param room:
        :param door: numpy.ndarray
        :param t_aset_0: Initial available safe egress time.
        :param interval: Interval for updating strategies
        :param neighbor_radius:
        :param neighborhood_size:
        """
        super().__init__()
        # TODO: Not include agent that have reached their goals
        # TODO: check if j not in players:
        # TODO: Update agents parameters by the new strategy

        self.simulation = simulation
        self.door = door
        if isinstance(room1, Polygon):
            vertices = np.asarray(room1.exterior)
        elif isinstance(room1, np.ndarray):
            vertices = room1
        else:
            raise Exception()

        self.room = Path(vertices)  # Polygon vertices

        # Game properties
        self.strategies = np.array((0, 1), dtype=np.int64)
        self.interval = interval
        self.t_aset_0 = t_aset_0
        self.t_aset = t_aset_0
        self.t_evac = np.zeros(self.simulation.agent.size, dtype=np.float64)
        self.radius = np.max(self.simulation.agent.radius)

        # Agent states
        self.strategy = np.ones(self.simulation.agent.size, dtype=np.int64)
        self.mask = np.ones(self.simulation.agent.size)

        # Set neigbourhood
        self.simulation.agent.neighbor_radius = neighbor_radius
        self.simulation.agent.neighborhood_size = neighborhood_size
        self.simulation.agent.reset_neighbor()

    def reset(self):
        self.t_evac[:] = 0

    def update(self):
        """Update strategies for all agents.

        :param dt: Timestep used by integrator to update simulation.
        """
        self.reset()

        # Indices of agents that are playing the game
        # Loop over agents and update strategies
        agent = self.simulation.agent
        self.mask = agent.active & self.room.contains_points(agent.position)
        indices = np.arange(agent.size)[self.mask]

        # Agents that are not playing anymore will be patient again
        self.strategy[self.mask ^ True] = 1

        self.t_aset = self.t_aset_0 - self.simulation.time_tot
        poisson_update(self.simulation.agent, indices, self.door,
                               self.radius, self.strategy, self.strategies, self.t_evac,
                               self.t_aset, self.interval, self.simulation.dt)

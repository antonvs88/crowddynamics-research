import numba
import numpy as np
from numba import float64, int64, boolean
from numba.types import UniTuple

from crowddynamics.core.vector2D import rotate270

spec_agent = (
    ("size", int64),
    ("ndim", int64),
    ("shape", UniTuple(int64, 2)),

    ("circular", boolean),
    ("three_circle", boolean),

    ("orientable", boolean),
    ("active", boolean[:]),
    ("in_room1", boolean[:]),
    ("goal_reached", boolean[:]),
    ("mass", float64[:, :]),
    ("radius", float64[:]),
    ("r_t", float64[:]),
    ("r_s", float64[:]),
    ("r_ts", float64[:]),

    ("position", float64[:, :]),
    ("velocity", float64[:, :]),
    ("target_velocity", float64[:, :]),
    ("target_direction", float64[:, :]),
    ("force", float64[:, :]),
    ("prev_force", float64[:, :]),
    ("prev_force_walls", float64[:, :]),
    ("prev_h", float64[:, :]),
    ("prev_n", float64[:, :, :]),
    ("prev_h_walls", float64[:, :]),
    ("contact_pressure", float64[:]),
    ("inertia_rot", float64[:]),
    ("angle", float64[:]),
    ("angular_velocity", float64[:]),
    ("target_angle", float64[:]),
    ("target_angular_velocity", float64[:]),
    ("torque", float64[:]),
    ("position_ls", float64[:, :]),
    ("position_rs", float64[:, :]),
    ("front", float64[:, :]),
)

spec_agent_motion = (
    ("tau_adj", float64[:]),
    ("tau_rot", float64),
    ("k_soc", float64[:]),
    ("tau_0", float64),
    ("mu", float64),
    ("kappa", float64),
    ("damping", float64),
    ("std_rand_force", float64[:]),
    ("std_rand_torque", float64),
    ("f_soc_ij_max", float64),
    ("f_soc_iw_max", float64),
    ("sight_soc", float64),
    ("sight_wall", float64),
    ("A_agent", float64[:]),
    ("B_agent", float64[:]),
    ("A_wall", float64[:]),
    ("B_wall", float64[:]),
    ("anisotropy_agent", float64),
    ("anisotropy_wall", float64),
)

spec_agent_neighbour = (
    ("neighbor_radius", float64),
    ("neighborhood_size", int64),
    ("neighbors", int64[:, :]),
    ("neighbor_distances", float64[:, :]),
    ("neighbor_distances_max", float64[:]),
)

spec_agent += spec_agent_motion + spec_agent_neighbour


@numba.jitclass(spec_agent)
class Agent(object):
    """Structure for agent parameters and variables."""

    def __init__(self, walls, size, mass, radius, ratio_rt, ratio_rs, ratio_ts,
                 inertia_rot, target_velocity, target_angular_velocity):

        # Array properties
        self.size = size  # Maximum number of agents
        self.ndim = 2     # 2-Dimensional space
        self.shape = (self.size, self.ndim)  # Shape of 2D arrays

        # Agent models (Only one can be active at time).
        self.circular = True
        self.three_circle = False

        # Flags
        self.orientable = self.three_circle  # Orientable has rotational motion
        self.active = np.zeros(size, np.bool8)  # Initialise agents as inactive
        self.in_room1 = np.ones(size, np.bool8)
        self.goal_reached = np.zeros(size, np.bool8)  # TODO: Deprecate

        # Constant properties
        self.radius = radius  # Total radius
        self.r_t = ratio_rt * radius  # Radius of torso
        self.r_s = ratio_rs * radius  # Radius of shoulders
        self.r_ts = ratio_ts * radius  # Distance from torso to shoulder
        self.mass = 80*np.ones((size, 1), dtype = float64)
        self.inertia_rot = inertia_rot  # Moment of inertia

        # Movement along x and y axis.
        self.position = np.zeros(self.shape)
        self.velocity = np.zeros(self.shape)
        self.target_velocity = 1*np.ones((size, 1), dtype = float64)
        self.target_direction = np.zeros(self.shape)
        self.force = np.zeros(self.shape)
        
        self.prev_force = np.zeros(self.shape)
        self.prev_force_walls = np.zeros((self.size, walls))
        self.contact_pressure = np.zeros(self.size)
        self.prev_n = np.zeros((self.size, self.size, self.ndim))
        self.prev_h = np.zeros((self.size, self.size))
        self.prev_h_walls = np.zeros((self.size, walls))

        # Rotational movement. Three circles agent model
        self.angle = np.zeros(self.size)
        self.angular_velocity = np.zeros(self.size)
        self.target_angle = np.zeros(self.size)
        self.target_angular_velocity = target_angular_velocity
        self.torque = np.zeros(self.size)

        self.position_ls = np.zeros(self.shape)  # Left shoulder
        self.position_rs = np.zeros(self.shape)  # Right shoulder
        self.front = np.zeros(self.shape)  # For plotting agents.
        self.update_shoulder_positions()

        # Motion related parameters
        self.tau_adj = np.zeros(self.size)  # Adjusting force
        self.tau_rot = 0.2  # Adjusting torque
        self.k_soc = np.zeros(self.size)  # Social force scaling
        self.tau_0 = 3  # Social force interaction time horizon. 3s was used by Karamouzas et al.
        self.mu = 1.2e5  # Body force counteracting body compression. 12e4 in FDS+Evac 1.2e5 helbing
        self.kappa = 2.4e5 # Tangential friction force. 4e4 in FDS+Evac 2.4e5 helbing
        self.damping = 500  # Damping force. 500 in FDS+Evac
        self.std_rand_force = 0.1*np.ones(self.size)  # Fluctuation force
        self.std_rand_torque = 0.1  # Fluctuation torque
        self.f_soc_ij_max = 2e3  # Truncation value for social force
        self.f_soc_iw_max = 2e3  # Truncation value for social force
        self.sight_soc = 3.0  # Interaction distance with other agents
        self.sight_wall = 3.0  # Interaction distance with walls

        # Motion related parameters unique to Helbing's model
        self.A_agent = np.zeros(self.size) # Helbing (2000) uses A_ij=2000
        self.B_agent = np.zeros(self.size) # Helbing (2000) uses B_ij=0.08
        self.A_wall = np.zeros(self.size) # Helbing (2000) uses A_iw=2000
        self.B_wall = np.zeros(self.size) # Helbing (2000) uses B_iw=0.08
        self.anisotropy_agent = 0.5
        self.anisotropy_wall = 0.2

        # Tracking neighboring agents. Neighbors contains the indices of the
        # neighboring agents. Negative value denotes missing value (if less than
        # neighborhood_size of neighbors).
        #
        # Make number of neighbors large enough to fit all agents within the given radius.
        # We could also change self.neighbors to be mutable.
        self.neighbor_radius = np.nan
        self.neighborhood_size = 100
        self.neighbors = np.ones((self.size, self.neighborhood_size), dtype=np.int64)
        self.neighbor_distances = np.zeros((self.size, self.neighborhood_size))
        self.neighbor_distances_max = np.zeros(self.size)
        self.reset_neighbor()

    def set_circular(self):
        self.circular = True
        self.three_circle = False
        self.orientable = self.three_circle

    def set_three_circle(self):
        self.circular = False
        self.three_circle = True
        self.orientable = self.three_circle

    def reset_motion(self):
        self.prev_force[:] = self.force[:]
        self.force[:] = 0
        self.torque[:] = 0
        self.contact_pressure[:] = 0

    def reset_neighbor(self):
        self.neighbors[:, :] = -1  # negative value denotes missing value
        self.neighbor_distances[:, :] = np.inf
        self.neighbor_distances_max[:] = np.inf

    def indices(self):
        """Indices of active agents."""
        # TODO: Other masks
        all_indices = np.arange(self.size)
        return all_indices[self.active]

    def update_shoulder_position(self, i):
        n = np.array((np.cos(self.angle[i]), np.sin(self.angle[i])))
        t = rotate270(n)
        offset = t * self.r_ts[i]
        self.position_ls[i] = self.position[i] - offset
        self.position_rs[i] = self.position[i] + offset
        self.front[i] = self.position[i] + n * self.r_t[i]

    def update_shoulder_positions(self):
        for i in self.indices():
            self.update_shoulder_position(i)

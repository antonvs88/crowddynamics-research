import numba
import numpy as np
from numba import f8

from .vector2D import truncate, length, dot2d

@numba.jit(nopython=True, nogil=True)
def force_helbing_anisotropic_circular(agent, i, j):
    """Social force between agent and agent (Helbing, 2000)."""
    r_ij = agent.radius[i] + agent.radius[j]  # sum of the agents' i and j radiis
    x_ij = agent.position[i] - agent.position[j]  # relative position vector
    d_ij = length(x_ij)  # distance between agents' i and j centres of mass
    n_ij = x_ij / d_ij  # normalized vector pointing from pedestrian j to i

    force = np.zeros(2), np.zeros(2)

    if length(agent.velocity[i]) != 0:
        e_i = agent.velocity[i] / length(agent.velocity[i])
        force[0][:] += agent.A_agent[i] * np.exp((r_ij - d_ij) / agent.B_agent[i]) * n_ij * (agent.anisotropy_agent +
                                                                                             (1-agent.anisotropy_agent)
                                                                                             * (1+dot2d(-n_ij,e_i))/2)
    if length(agent.velocity[j]) != 0:
        e_j = agent.velocity[j] / length(agent.velocity[j])
        force[1][:] += agent.A_agent[j] * np.exp((r_ij - d_ij) / agent.B_agent[j]) * (-n_ij) * (agent.anisotropy_agent +
                                                                                            (1-agent.anisotropy_agent)
                                                                                            * (1 + dot2d(n_ij,e_j)) / 2)
    return force

@numba.jit(nopython=True, nogil=True)
def force_helbing_anisotropic_linear_wall(d, n, i, w, agent, wall):
    """"Social force between agent and wall (Helbing, 2000)."""
    force = np.zeros(2)

    if length(agent.velocity[i]) != 0:
        e_i = agent.velocity[i] / length(agent.velocity[i])
        force[:] = agent.A_wall[i] * np.exp((agent.radius[i] - d) / agent.B_wall[i]) * n * (agent.anisotropy_wall +
                                                                                            (1-agent.anisotropy_wall) *
                                                                                            (1 + dot2d(-n, e_i)) / 2)
    return force

# The following functions are based on Helbing's Nature article (2000)

@numba.jit(nopython=True, nogil=True)
def force_helbing_circular(agent, i, j):
    """Social force between agent and agent (Helbing, 2000)."""
    r_ij = agent.radius[i] + agent.radius[j] # sum of the agents' i and j radiis
    x_ij = agent.position[i] - agent.position[j] # relative position vector
    d_ij = length(x_ij)# distance between agents' i and j centres of mass
    n_ij = x_ij/d_ij # normalized vector pointing from pedestrian j to i

    force = np.zeros(2), np.zeros(2)

    # force[0][:] is the force affecting agent i, and force[1][:] is the force affecting agent j.

    force[0][:] += agent.A_agent[i]*np.exp((r_ij-d_ij)/agent.B_agent[i])*n_ij
    force[1][:] += agent.A_agent[j]*np.exp((r_ij-d_ij)/agent.B_agent[j])*(-n_ij)
    return force

@numba.jit(nopython=True, nogil=True)
def force_helbing_linear_wall(d, n, i, w, agent, wall):
    """"Social force between agent and wall (Helbing, 2000)."""
    force = np.zeros(2)
    force[:] = agent.A_wall[i]*np.exp((agent.radius[i] - d)/agent.B_wall[i])*n
    return force

@numba.jit(f8[:](f8, f8, f8[:], f8[:], f8[:], f8, f8), nopython=True, nogil=True)
def force_helbing_contact_agent_agent(r_tot, d, n, v, t, mu, kappa):
    """Frictional contact force between agent and agent (Helbing, 2000)."""
    return mu * (r_tot - d) * n + kappa * (r_tot -d) * dot2d(v, t) * t

@numba.jit(f8[:](f8, f8, f8[:], f8[:], f8[:], f8, f8), nopython=True, nogil=True)
def force_helbing_contact_agent_wall(r_tot, d, n, v, t, mu, kappa):
    """Frictional contact force between agent and wall (Helbing, 2000)."""
    return mu * (r_tot - d) * n - kappa * (r_tot - d) * dot2d(v, t) * t

@numba.jit(f8[:](f8, f8, f8[:], f8[:], f8[:], f8, f8, f8), nopython=True, nogil=True)
def force_langston_contact_agent_agent(r_tot, d, n, v, t, mu, kappa, damping):
    """Frictional contact force between agent and agent (Helbing, 2000)."""
    return mu * (r_tot - d) * n + kappa * (r_tot -d) * dot2d(v, t) * t + damping * dot2d(v, n) * n

@numba.jit(f8[:](f8, f8, f8[:], f8[:], f8[:], f8, f8, f8), nopython=True, nogil=True)
def force_langston_contact_agent_wall(r_tot, d, n, v, t, mu, kappa, damping):
    """Frictional contact force between agent and wall (Helbing, 2000)."""
    return mu * (r_tot - d) * n - kappa * (r_tot - d) * dot2d(v, t) * t + damping * dot2d(v, n) * n

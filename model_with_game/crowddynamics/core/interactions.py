import numba
import numpy as np

#from .motion import force_contact
from .power_law import force_social_circular, force_social_three_circle, \
    force_social_linear_wall
from .helbing import force_helbing_circular, force_helbing_linear_wall, force_helbing_contact_agent_agent, \
    force_helbing_contact_agent_wall, force_langston_contact_agent_agent, force_langston_contact_agent_wall, \
    force_helbing_anisotropic_circular, force_helbing_anisotropic_linear_wall
from .vector2D import length, rotate270, cross2d, dot2d


@numba.jit(nopython=True, nogil=True)
def agent_agent(agent):
    # n - 1 + n - 2 + ... + 1 = n^2 / 2 in O(n^2)
    ind = agent.indices()
    for l, i in enumerate(ind[:-1]):
        for j in ind[l + 1:]:
            agent_agent_interaction(i, j, agent)


@numba.jit(nopython=True, nogil=True)
def agent_wall(agent, wall):
    ind = agent.indices()
    for i in ind:
        for w in range(wall.size):
            agent_wall_interaction(i, w, agent, wall)


@numba.jit(nopython=True, nogil=True)
def agent_agent_distance_three_circle(agent, i, j):
    """Distance between two three-circle models.

    :param agent:
    :param i:
    :param j:
    :return:
    """
    # Positions: center, left, right
    x_i = (agent.position[i], agent.position_ls[i], agent.position_rs[i])
    x_j = (agent.position[j], agent.position_ls[j], agent.position_rs[j])

    # Radii of torso and shoulders
    r_i = (agent.r_t[i], agent.r_s[i], agent.r_s[i])
    r_j = (agent.r_t[j], agent.r_s[j], agent.r_s[j])

    # Minimizing values
    positions = np.zeros(agent.shape[1]), np.zeros(agent.shape[1])  #
    radius = (0.0, 0.0)  # Radius
    relative_distance = np.nan  # Minimum relative distance distance
    normal = np.zeros(agent.shape[1])  # Unit vector of x_rel

    for xi, ri in zip(x_i, r_i):
        for xj, rj in zip(x_j, r_j):
            x = xi - xj
            d = length(x)
            r_tot = (ri + rj)
            h = d - r_tot
            if h < relative_distance or np.isnan(relative_distance):
                relative_distance = h
                radius = ri, rj
                normal = x / d
                positions = xi, xj

    r_moment_i = positions[0] + radius[0] * normal - agent.position[i]
    r_moment_j = positions[1] - radius[1] * normal - agent.position[j]

    return normal, relative_distance, r_moment_i, r_moment_j


@numba.jit(nopython=True, nogil=True)
def agent_wall_distance(agent, wall, i, w):
    """Distance between three-circle model and a line.

    :param agent:
    :param wall:
    :param i:
    :param w:
    :return:
    """
    x_i = (agent.position[i], agent.position_ls[i], agent.position_rs[i])
    r_i = (agent.r_t[i], agent.r_s[i], agent.r_s[i])

    relative_distance = np.nan
    position = np.zeros(2)
    normal = np.zeros(2)
    radius = 0.0

    for xi, ri in zip(x_i, r_i):
        d, n = wall.distance_with_normal(w, xi)
        h = d - ri
        if h < relative_distance or np.isnan(relative_distance):
            relative_distance = h
            position = xi
            radius = ri
            normal = n

    r_moment_i = position - radius * normal - agent.position[i]

    return relative_distance, normal, r_moment_i


@numba.jit(nopython=True, nogil=True)
def agent_agent_interaction(i, j, agent):
    # Function params
    x = agent.position[i] - agent.position[j]  # Relative positions
    d = length(x)  # Distance
    r_tot = agent.radius[i] + agent.radius[j]  # Total radius
    h = d - r_tot  # Relative distance
    n = x / d  # Normal vector

    # Agent sees the other agent
    if d <= agent.sight_soc:
        if agent.three_circle:
            # Three circle model
            # TODO: Merge functions
            n, h, r_moment_i, r_moment_j = agent_agent_distance_three_circle(agent, i, j)
            force_i, force_j = force_social_three_circle(agent, i, j)
        else:
            # Circular model
            # Social force between agent and agent (Helbing, 2000).
            force_i, force_j = force_helbing_circular(agent, i, j)
            r_moment_i, r_moment_j = np.zeros(2), np.zeros(2)

        # Physical contact
        if h < 0:
            t = rotate270(n)  # Tangent vector
            v = agent.velocity[j] - agent.velocity[i]  # Relative velocity
            # Contact force between agent and agent (Helbing, 2000).
            force_c = force_langston_contact_agent_agent(r_tot, d, n, v, t, agent.mu, agent.kappa, agent.damping)
            # We assume here that the contact force is symmetrical for agent i and agent j.
            force_i += force_c
            force_j -= force_c

        agent.force[i] += force_i
        agent.force[j] += force_j
        if agent.orientable:
            agent.torque[i] += cross2d(r_moment_i, force_i)
            agent.torque[j] += cross2d(r_moment_j, force_j)

    if agent.neighbor_radius > 0 and h < agent.neighbor_radius:
        ind = np.argmin(agent.neighbors[i, :])
        agent.neighbors[i, ind] = j

        ind = np.argmin(agent.neighbors[j, :])
        agent.neighbors[j, ind] = i
    
    # Calculate contact pressure experienced by the agents. If agents i and j where in contact with each other in last iteration,
    # add the magnitude of the component of agent j's previous resultant force pointing towards agent i (and vice versa) to the
    # contact pressure term. This is achieved by taking the absolute value of the dot product of agent j's previous resultant
    # force and the previous normal vector pointing from agent j to i (and vice versa).
    if agent.prev_h[i, j] < 0:
        agent.contact_pressure[i] += np.abs(dot2d(agent.prev_force[j], agent.prev_n[i, j, :]))
        agent.contact_pressure[j] += np.abs(dot2d(agent.prev_force[i], agent.prev_n[j, i, :]))
    # Save the relative distance and normal vectors for the next iteration
    agent.prev_h[i, j] = h
    agent.prev_h[j, i] = h
    agent.prev_n[i, j, 0] = n[0]
    agent.prev_n[i, j, 1] = n[1]
    agent.prev_n[j, i, 0] = -n[0]
    agent.prev_n[j, i, 1] = -n[1]


@numba.jit(nopython=True, nogil=True)
def agent_wall_interaction(i, w, agent, wall):
    # Function params
    x = agent.position[i]
    r_tot = agent.radius[i]
    d, n = wall.distance_with_normal(w, x)
    h = d - r_tot

    if d <= agent.sight_wall:
        if agent.three_circle:
            h, n, r_moment_i = agent_wall_distance(agent, wall, i, w)
            r_moment_i = np.zeros(2)
            force = force_social_linear_wall(i, w, agent, wall)
        else:
            # Circular model
            r_moment_i = np.zeros(2)
            # Social force between agent and wall (Helbing, 2000).
            force = force_helbing_linear_wall(d, n, i, w, agent, wall)

        if h < 0:
            t = rotate270(n)  # Tangent
            # Contact force between agent and wall (Helbing, 2000).
            force_c = force_langston_contact_agent_wall(r_tot, d, n, agent.velocity[i], t, agent.mu, agent.kappa,
                                                        agent.damping)
            force += force_c
            agent.prev_force_walls[i, w] = length(force_c)

        agent.force[i] += force
        if agent.orientable:
            agent.torque[i] += cross2d(r_moment_i, force)

    if agent.prev_h_walls[i, w] < 0:
        agent.contact_pressure[i] += agent.prev_force_walls[i, w]
    agent.prev_h_walls[i, w] = h

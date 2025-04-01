import numpy as np

# Constants
G = 6.67430e-11
c = 3.0e8
AU = 1.496e11
year = 3.154e7
M_sun = 1.989e30
M_mercury = 3.3011e23
a = 0.387 * AU
e = 0.2056
r0 = a * (1 - e)
v0 = np.sqrt(G * M_sun * (1 + e) / (a * (1 - e)))
dt = 30.0
sim_years = 5.0
num_steps = int(sim_years * year / dt)
masses = np.array([M_mercury, M_sun])
EPSILON = 1e-20

def compute_four_accelerations(pos, vel, masses):
    n = len(masses)
    acc = np.zeros((n, 4))
    R_ij = pos[:, None, :] - pos  # Position differences (n x n x 4)
    r_4d = np.sqrt(np.abs(np.einsum('ijk,kl,ijl->ij', R_ij, eta, R_ij))) + EPSILON  # 4D distances (n x n)

    for i in range(n):
        r_vec = pos[i] - pos  # Position vector relative to particle i (n x 4)
        r_spatial = np.linalg.norm(r_vec, axis=1) + EPSILON  # Spatial distances (n)
        v_rel = vel[i] - vel  # Relative velocities (n x 4)
        v_sq = np.sum(v_rel**2, axis=1)  # v_rel^2 (n)
        r_dot_v = np.sum(r_vec * v_rel, axis=1)  # r_ij · v_rel (n)
        mass_prod = masses[i] * masses  # m_i * m_j (n)

        # Updated analytical form of F_i^mu (Equation 39 from Section 7)
        # The extra factors of 4 have been removed per the new derivation.
        F_total = (
            -G * mass_prod[:, None] * r_vec / r_spatial[:, None]**3  # Newtonian term
            + (G * mass_prod[:, None] / (c**2 * r_spatial[:, None]**3)) * (
                (G * masses / r_spatial - v_sq)[:, None] * r_vec  # Position-dependent relativistic term
                + r_dot_v[:, None] * v_rel  # Velocity-dependent relativistic term
            )
        )

        # Mask to exclude self-interaction (i != j)
        mask = (np.arange(n) != i)[:, None]
        F_total = np.sum(F_total * mask, axis=0)  # Sum over j != i (4)

        acc[i] = F_total / masses[i]  # Acceleration (4)
    return acc

    
def update_particles(pos, vel, masses, dt):
    acc = compute_four_accelerations(pos, vel, masses)
    vel += acc * dt
    pos += vel * dt
    return pos, vel

# Initialization and simulation loop
n = len(masses)
pos = np.zeros((n, 4))
vel = np.zeros((n, 4))
history = np.zeros((num_steps, n, 4))
radii = np.zeros(num_steps)
pos[0] = [0, r0, 0, 0]
v0_3d = np.array([0, v0, 0])
gamma_0 = 1 / np.sqrt(1 - (v0 / c)**2)
vel[0] = [gamma_0 * c, *v0_3d]
pos[1] = [0, 0, 0, 0]
v1_3d = np.array([0, - (M_mercury / M_sun) * v0 / 2, 0])
gamma_1 = 1 / np.sqrt(1 - (v0 / c)**2)
vel[1] = [gamma_1 * c, *v1_3d]
history[0] = pos.copy()
radii[0] = np.sqrt(np.abs(np.einsum('i,ij,j->', pos[0] - pos[1], eta, pos[0] - pos[1])))
angles = []
times = []
print("Perihelia (time, radius, angle):");
for i in range(1, num_steps):
    pos, vel = update_particles(pos, vel, masses, dt)
    history[i] = pos.copy()
    eta = np.eye(4)
    radii[i] = np.sqrt(np.abs(np.einsum('i,ij,j->', pos[0] - pos[1], eta, pos[0] - pos[1])))
    if i > 1 and radii[i] > radii[i-1] and radii[i-1] < radii[i-2]:
        t_vals = np.array([i-2, i-1, i]) * dt
        r_vals = radii[i-2:i+1]
        coeffs = np.polyfit(t_vals, r_vals, 2)
        t_min = -coeffs[1] / (2 * coeffs[0])
        t = t_min/year
        if len(times) == 0 or (times[-1] + 0.4) < t:
            alpha = (t_min - t_vals[1]) / dt
            pos_rel_prev = history[i-1, 0, 1:] - history[i-1, 1, 1:]
            pos_rel_curr = history[i, 0, 1:] - history[i, 1, 1:]
            pos_min = (1 - alpha) * pos_rel_prev + alpha * pos_rel_curr
            angle = np.arctan2(pos_min[1], pos_min[0])
            angles.append(angle)
            times.append(t)
            print(f"t={t_min/year:.6f} years, r={r_vals.min():.2f}, angle={angle:.10f}")
if len(angles) > 1:
    angles = np.unwrap(angles)
    delta_phi = angles[-1] - angles[0]
    orbits = len(angles) - 1
    precession_per_orbit = delta_phi / orbits
    orbital_period = 87.969 / 365.25
    orbits_per_century = 100 / orbital_period
    precession_arcsec = precession_per_orbit * orbits_per_century * (180 / np.pi) * 3600
    print(f"Orbits detected: {orbits}")
    print(f"Total precession: {delta_phi:.10f} rad")
    print(f"Precession per orbit: {precession_per_orbit:.10f} rad")
    print(f"Precession: {precession_arcsec:.2f} arcsec/century")
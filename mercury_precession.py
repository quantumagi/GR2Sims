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
    """
    Compute four-accelerations in pre-projection 4D Euclidean space.
    
    Parameters:
    - pos: (n, 4) array of positions [x0, x1, x2, x3]
    - vel: (n, 4) array of velocities [v0, v1, v2, v3]
    - masses: (n,) array of particle masses
    
    Returns:
    - acc: (n, 4) array of four-accelerations
    """
    n = len(masses)
    acc = np.zeros((n, 4))  # Four-acceleration array

    for i in range(n):
        F_total = np.zeros(4)  # Total four-force on particle i
        for j in range(n):
            if i != j:
                # Spatial separation (x1, x2, x3 components)
                r_vec_spatial = pos[i, 1:] - pos[j, 1:]
                r_spatial = np.sqrt(np.sum(r_vec_spatial**2)) + EPSILON
                
                # Newtonian force (spatial components only)
                F_newton = -G * masses[i] * masses[j] * r_vec_spatial / r_spatial**3
                
                # Optional: Simplified post-Newtonian correction
                v_rel = vel[i] - vel[j]
                v_sq = np.sum(v_rel**2)  # Euclidean norm in 4D
                r_dot_v = np.sum(r_vec_spatial * v_rel[1:])  # Spatial dot product
                F_pn = (G * masses[i] * masses[j] / (c**2 * r_spatial**3)) * (
                    (4 * G * masses[j] / r_spatial - v_sq) * r_vec_spatial
                    + 4 * r_dot_v * v_rel[1:]
                )
                
                # Add to spatial components (F^0 = 0)
                F_total[1:] += F_newton + F_pn
        
        # Four-acceleration: a^\mu = f^\mu / m
        acc[i] = F_total / masses[i]
    
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
        alpha = (t_min - t_vals[1]) / dt
        pos_rel_prev = history[i-1, 0, 1:] - history[i-1, 1, 1:]
        pos_rel_curr = history[i, 0, 1:] - history[i, 1, 1:]
        pos_min = (1 - alpha) * pos_rel_prev + alpha * pos_rel_curr
        angle = np.arctan2(pos_min[1], pos_min[0])
        angles.append(angle)
        times.append(t_min / year)
        print(f"Perihelion at t={t_min/year:.6f} years, r={r_vals.min():.10f}, angle={angle:.10f}")
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
    print("Perihelion angles:", angles)
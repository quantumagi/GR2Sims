# cosmic_expansion.ipynb (or save as .py for VS Code Jupyter)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # For 3D plots
from IPython.display import clear_output, display

# Physics constants
c = 299792458  # Speed of light in m/s
G = 6.6740105e-11  # Gravitational constant in m^3 kg^-1 s^-2
secyear = 31536000  # Seconds per year
t0 = secyear * 14.37e9  # Approximate age of the universe in seconds (14.37 Gyr)

# Simulation constants
num_particles = 250  # Number of particles
iterations = 1000  # Reduced for performance in notebook (adjust as needed)
R = 1e6  # Initial confined region radius in meters
mass = 1  # Mass scale (arbitrary in Planck units or kg, adjust as needed)
EPSILON = 1e-10

# Initialize particle positions randomly within a 4D hypercube
np.random.seed(42)  # For reproducibility
positions_4d = np.random.uniform(-R, R, (num_particles, 4))

# Initialize velocity components randomly and normalize to maintain magnitude c
velocities_4d = np.random.uniform(-1e1, 1e1, (num_particles, 4))
velocities_4d = velocities_4d / np.linalg.norm(velocities_4d, axis=1)[:, None] * c

# Set up the figure and subplots for visualization
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
axs[0, 0] = fig.add_subplot(221, projection='3d')
axs[0, 0].set_title('Particle Positions and Velocity Directions')
axs[0, 0].set_xlabel('X axis')
axs[0, 0].set_ylabel('Y axis')
axs[0, 0].set_zlabel('Z axis')

axs[0, 1].set_title('Outward Velocity vs Radial Distance')
axs[0, 1].set_xlabel('Radial Distance from Center of Mass')
axs[0, 1].set_ylabel('Outward Component of Velocity')

axs[1, 0].set_title('Temporal vs Spatial Separation')
axs[1, 0].set_xlabel('Spatial Separation (m)')
axs[1, 0].set_ylabel('Temporal Separation (s)')

axs[1, 1].set_title('Average Radial Distance Over Time')
axs[1, 1].set_xlabel('Time (Years)')
axs[1, 1].set_ylabel('Average Radial Distance (m)')
plt.tight_layout()

def compute_four_accelerations(pos, vel, masses):
    n = len(masses)
    acc = np.zeros((n, 4))

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

def derive_spatio_temporal_data(positions, velocities_4d):
    spatial_separations = []
    temporal_separations = []
    for i in range(num_particles):
        for j in range(num_particles):
            if i == j:
                continue
            relative_position = positions[j] - positions[i]
            velocity_dir = velocities_4d[i] / np.linalg.norm(velocities_4d[i])
            temporal_sep = np.dot(relative_position, velocity_dir)
            projection_matrix = np.eye(4) - np.outer(velocity_dir, velocity_dir)
            spatial_sep = np.linalg.norm(projection_matrix @ relative_position)
            spatial_separations.append(spatial_sep)
            temporal_separations.append(temporal_sep)
    return spatial_separations, temporal_separations

def update_plot(positions, velocities_4d, Rs, spatial_separations, temporal_separations, curvatures):
    ax_3d = axs[0, 0]
    ax_3d.clear()
    
    xs = positions[:, 0]
    ys = positions[:, 1]
    zs = positions[:, 2]
    ws = positions[:, 3]
    
    r = np.linalg.norm(positions, axis=1)
    nonzero = r > 1e-14
    chi = np.zeros_like(r)
    theta = np.zeros_like(r)
    phi_angle = np.zeros_like(r)
    
    chi[nonzero] = np.arccos(ws[nonzero] / r[nonzero])
    sin_chi = np.sin(chi[nonzero])
    theta[nonzero] = np.arccos(np.clip(zs[nonzero] / (r[nonzero]*sin_chi), -1.0, 1.0))
    phi_angle[nonzero] = np.arctan2(ys[nonzero], xs[nonzero])
    
    X = r * np.sin(theta) * np.cos(phi_angle)
    Y = r * np.sin(theta) * np.sin(phi_angle)
    Z = r * np.cos(theta)
    
    if np.any(nonzero):
        chi_min, chi_max = chi[nonzero].min(), chi[nonzero].max()
        chi_norm = np.zeros_like(chi)
        if chi_max > chi_min:
            chi_norm[nonzero] = (chi[nonzero] - chi_min) / (chi_max - chi_min)
        else:
            chi_norm[nonzero] = 0.5
    else:
        chi_norm = np.full_like(chi, 0.5)
    
    colors = plt.cm.viridis(chi_norm)
    
    speed = np.linalg.norm(velocities_4d, axis=1)
    vxs = velocities_4d[:, 0]
    vys = velocities_4d[:, 1]
    vzs = velocities_4d[:, 2]
    
    spread = np.max(np.abs(np.column_stack((X, Y, Z)))) * 1.1
    ax_3d.set_xlim([-spread, spread])
    ax_3d.set_ylim([-spread, spread])
    ax_3d.set_zlim([-spread, spread])
    ax_3d.set_xlabel('X axis')
    ax_3d.set_ylabel('Y axis')
    ax_3d.set_zlabel('Z axis')
    ax_3d.set_title('4D Points: (r,θ,φ) → 3D, color=χ; Velocity quivers shown')
    ax_3d.scatter(X, Y, Z, c=colors, s=10)
    for i in range(num_particles):
        length = r[i] / 10 if r[i] != 0 else 0.1
        ax_3d.quiver(X[i], Y[i], Z[i], vxs[i], vys[i], vzs[i], length=length, color='r', normalize=True)
    
    ax_2d = axs[0, 1]
    ax_2d.clear()
    ax_2d.set_title('Outward Velocity vs Radial Distance')
    center_of_mass = np.mean(positions, axis=0)
    radial_vectors = positions - center_of_mass
    radial_distances = np.linalg.norm(radial_vectors, axis=1, keepdims=True)
    radial_unit_vectors = radial_vectors / (radial_distances + 1e-12)
    outward_components = np.einsum('ij,ij->i', velocities_4d, radial_unit_vectors)
    ax_2d.scatter(radial_distances, outward_components, color='blue', s=5, alpha=0.5)
    ax_2d.axhline(0, color='black', linestyle='--', linewidth=1)
    ax_2d.axvline(Rs, color='red', linestyle='--', linewidth=1, label=f'R_s={Rs:.2e}')
    ax_2d.legend(loc="upper left")
    ax_2d.set_xlabel('Radial Distance from Center of Mass')
    ax_2d.set_ylabel('Outward Component of Velocity')
    ax_2d.set_xlim(left=0)
    
    ax_3 = axs[1, 0]
    ax_3.clear()
    ax_3.set_title('Temporal vs Spatial Separation')
    ax_3.scatter(spatial_separations, temporal_separations, s=3, alpha=0.7, color='green', label='Data')
    ax_3.set_xlabel('Spatial Separation (m)')
    ax_3.set_ylabel('Temporal Separation (s)')
    ax_3.legend()

    ax_4 = axs[1, 1]
    ax_4.clear()
    ax_4.set_title('Average Radial Distance Over Time')
    ax_4.set_xlabel('Time (Years)')
    ax_4.set_ylabel('Average Radial Distance (m)')
    if len(curvatures) > 0:
        ax_4.plot(*zip(*curvatures), color='blue', label=f'Avg: {curvatures[-1][1]:.2e} m')
        ax_4.legend(loc='upper left')

    clear_output(wait=True)
    display(fig)
    plt.pause(0.001)  # Brief pause for display in VS Code Jupyter

# Main simulation loop with animation
t = 0.0001
R_over_M = 2 * G / c**2
M = R / R_over_M
curvatures = []

for _ in range(iterations):
    if t >= t0:
        break
    
    year = t / secyear
    R = R_over_M * M
    spatial_separations, temporal_separations = derive_spatio_temporal_data(positions_4d, velocities_4d)
    
    center_of_mass = np.mean(positions_4d, axis=0)
    radial_distances = np.linalg.norm(positions_4d - center_of_mass, axis=1)
    avg_radial_distance = np.mean(radial_distances)
    curvatures.append((year, avg_radial_distance))
    
    update_plot(positions_4d, velocities_4d, R, spatial_separations, temporal_separations, curvatures)
    
    dM = M * (0.001 if t / t0 > 0.9 else 0.05)
    dt = (2 * G / c**3) * dM
    
    positions_4d, velocities_4d = update_particles(positions_4d, velocities_4d, np.array([mass] * num_particles), dt)
    M += dM
    t += dt

plt.close(fig)
print("Simulation completed.")
# cosmic_distance.ipynb (or save as .py for VS Code Jupyter)
import requests
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from IPython.display import clear_output, display

# Constants
c = 299792.458  # Speed of light in km/s
H0 = 68.1  # Hubble constant in km/s/Mpc
M = -19.3  # Absolute magnitude of Type Ia supernovae

# Model distance modulus function
def distance_modulus_model(z):
    d_L = (c / H0) * (1 + z) * np.log(1 + z)  # H(z) = H0 (1 + z)
    mu = 5 * np.log10(d_L) + 25
    return mu

# LambdaCDM model function (for comparison)
def integrand(z, Omega_m, Omega_Lambda):
    return 1.0 / np.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda)

def distance_modulus_LCDM(z):
    D_C, _ = quad(integrand, 0, z, args=(0.3, 0.7))
    D_C *= (c / 73.9)  # H0 for LambdaCDM
    d_L = (1 + z) * D_C
    mu_LCDM = 5 * np.log10(d_L) + 25
    return mu_LCDM

# Fetch Pantheon dataset
url = 'https://raw.githubusercontent.com/dscolnic/Pantheon/master/lcparam_full_long.txt'
response = requests.get(url)
data_lines = response.content.decode('utf-8').splitlines()

# Parse dataset
z_obs, mB_obs, mB_obs_error = [], [], []
for line in data_lines[1:]:
    parts = line.split()
    z_obs.append(float(parts[1]))    
    mB_obs.append(float(parts[4]))    
    mB_obs_error.append(float(parts[5]))

z_obs, mB_obs, mB_obs_error = np.array(z_obs), np.array(mB_obs), np.array(mB_obs_error)

# Convert apparent magnitudes to distance moduli
mu_obs = mB_obs - M
mu_obs_error = mB_obs_error

# Define redshift bins
bins = np.arange(0, 2.5, 0.05)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Cluster data
z_cluster, mu_cluster, mu_error_cluster = [], [], []
for i in range(len(bins) - 1):
    mask = (z_obs >= bins[i]) & (z_obs < bins[i + 1])
    if np.any(mask):
        z_cluster.append(bin_centers[i])
        mu_cluster.append(np.mean(mu_obs[mask]))
        mu_error_cluster.append(np.std(mu_obs[mask]))

# Compute theoretical distance moduli
mu_model = np.array([distance_modulus_model(z) for z in z_cluster])
mu_LCDM = np.array([distance_modulus_LCDM(z) for z in z_cluster])

# Plot results interactively
fig = plt.figure(figsize=(12, 8))
plt.errorbar(z_cluster, mu_cluster, yerr=mu_error_cluster, fmt='o', label='Observed (Cluster)')
plt.plot(z_cluster, mu_model, label='GR² Model: H(z) = H0 (1 + z)', linestyle='--', color='orange')
plt.plot(z_cluster, mu_LCDM, label='LambdaCDM Model', linestyle='--', color='green')
plt.xlabel('Redshift z')
plt.ylabel('Distance Modulus \(\mu\)')
plt.legend()
plt.title('Distance Modulus Comparison for GR² and LambdaCDM')

clear_output(wait=True)
display(fig)
plt.pause(30)  # Brief pause for display in VS Code Jupyter

plt.close(fig)
print("Plot completed and displayed interactively.")
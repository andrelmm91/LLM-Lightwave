import numpy as np
from scipy.linalg import lstsq

# Parameters
N = 100000  # Scalable to 100k
M = 5       # Steps (layers)

# Initialize
amplitudes = np.random.uniform(0, 1, N)
phases = np.random.uniform(-np.pi, np.pi, N)
initial_z = amplitudes * np.exp(1j * phases)
u = np.zeros((M+1, N), dtype=complex)
v = np.zeros((M+1, N), dtype=complex)
u[0, :] = initial_z.real
v[0, :] = initial_z.imag

# Evolve vectorized
for m in range(M):
    # [Shifts with absorbing boundaries]
    
    # Phase/amplitude modulation
    phase_diff_u = np.angle(u_left + 1j * v_left) - np.angle(u[m] + 1j * v[m])
    amp_ratio_u = np.abs(u_left + 1j * v_left) / (np.abs(u[m] + 1j * v[m]) + 1e-8)
    mod_factor_u = np.tanh(phase_diff_u) * amp_ratio_u
    
    # [Similar for v]
    
    # Updates
    interference_u = 0.1 * mod_factor_u * (u_left + 1j * v_left)
    u[m+1, :] = u[m, :] + interference_u
    # [Similar for v]
    
    # Nonlinearity
    u[m+1, :] = np.tanh(u[m+1, :].real) + 1j * np.tanh(u[m+1, :].imag)
    # [For v]
    
    # Normalization
    z = u[m+1, :] + 1j * v[m+1, :]
    max_int = np.max(np.abs(z)**2)
    if max_int > 0:
        scale = 1.0 / np.sqrt(max_int)
        u[m+1, :] *= scale
        v[m+1, :] *= scale

# Final
final_z = u[M, :] + 1j * v[M, :]
intensities = np.abs(final_z)**2
phases_out = np.angle(final_z)

# Sample (for large N, mean stats)
print("Mean initial intensity:", np.mean(np.abs(initial_z)**2))
print("Mean final intensity:", np.mean(intensities))
print("Sample final phases:", phases_out[:5])
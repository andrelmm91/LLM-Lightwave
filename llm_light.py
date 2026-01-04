import numpy as np

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
    u_rolled = np.roll(u[m, :], 1)   # For x-1: roll brings x-1 to x
    v_rolled = np.roll(v[m, :], 1)
    u[m+1, :] = 1/np.sqrt(2) * (u_rolled + 1j * v_rolled)
    
    v_rolled_right = np.roll(v[m, :], -1)  # For x+1: roll -1 brings x+1 to x
    u_rolled_right = np.roll(u[m, :], -1)
    v[m+1, :] = 1/np.sqrt(2) * (v_rolled_right + 1j * u_rolled_right)

# Final
final_z = u[M, :] + 1j * v[M, :]
intensities = np.abs(final_z)**2
phases_out = np.angle(final_z)

# Sample (for large N, mean stats)
print("Mean initial intensity:", np.mean(np.abs(initial_z)**2))
print("Mean final intensity:", np.mean(intensities))
print("Sample final phases:", phases_out[:5])
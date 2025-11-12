import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Lorenz system equations
def lorenz(t, state, sigma=10, beta=8/3, rho=28):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Simulate the Lorenz system
def generate_lorenz_data(t_max=100, dt=0.01, initial_state=[1.0, 0.0, 0.0]):
    t_span = (0, t_max)
    t_eval = np.arange(0, t_max, dt)
    sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval, method='RK45')
    return sol.t, sol.y[0]  # Return time and x-coordinate

# Generate Lorenz time series
t, x = generate_lorenz_data()

np.random.seed(42)
n = 10000  # Number of time series entries
phi = 0.7  # AR(1) coefficient
sigma = 1.0  # Standard deviation of noise
y = np.zeros(n)
epsilon = np.random.normal(0, sigma, n)

# Generate AR(1) time series
y[0] = epsilon[0]
for t_idx in range(1, n):
    y[t_idx] = phi * y[t_idx-1] + epsilon[t_idx]

#t, x = np.linspace(0, 9999, 10000), y

# Fourier Transform of the time series
X = np.fft.fft(x)

# Extract Fourier phases
phases_original = np.angle(X)

# Phase randomization (generate surrogate)
random_phases = np.random.uniform(-np.pi, np.pi, size=X.shape)
X_randomized = np.abs(X) * np.exp(1j * random_phases)
phases_surrogate = np.angle(X_randomized)

# Define phase lag Δ
delta = 1  # Adjust for different phase relationships

# Generate phase maps: φ(k) vs. φ(k + Δ)
phi_k = phases_original[:-delta]
phi_k_delta = phases_original[delta:]

phi_k_surr = phases_surrogate[:-delta]
phi_k_delta_surr = phases_surrogate[delta:]

# Plot Phase Maps
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(phi_k, phi_k_delta, s=1, color='blue')
plt.title(f'Phase Map (Original) - Δ = {delta}')
plt.xlabel(r'$\phi(k)$')
plt.ylabel(r'$\phi(k + \Delta)$')

plt.subplot(1, 2, 2)
plt.scatter(phi_k_surr, phi_k_delta_surr, s=1, color='orange')
plt.title(f'Phase Map (Surrogate) - Δ = {delta}')
plt.xlabel(r'$\phi(k)$')
plt.ylabel(r'$\phi(k + \Delta)$')

plt.tight_layout()
plt.show()

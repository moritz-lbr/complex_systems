import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.stats import norm
from scipy.integrate import solve_ivp


# Define Lorenz system
def lorenz(t, state, sigma=10, beta=8 / 3, rho=28):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


# Solve Lorenz system
T = 100
dt = 0.01
t = np.arange(0, T, dt)
sol = solve_ivp(lorenz, [0, T], [1, 1, 1], t_eval=t)
x_orig = sol.y[0]  # Use only the x-component


# AAFT surrogate generation
def aaft_surrogate(x):
    # Step 1: Rank-order remap x to a Gaussian distribution
    sorted_x = np.sort(x)
    gaussian_x = np.random.randn(len(x))
    gaussian_x.sort()
    rank_indices = np.argsort(np.argsort(x))  # Get ranks of original data
    x_gaussian = gaussian_x[rank_indices]

    # Step 2: Fourier Transform and phase randomization
    Xf = fft(x_gaussian)
    random_phases = np.exp(1j * np.random.uniform(0, 2 * np.pi, len(x)))
    Xf_new = np.abs(Xf) * random_phases
    x_shuffled = np.real(ifft(Xf_new))

    # Step 3: Rank-order remap back to the original data distribution
    x_surrogate = np.zeros_like(x)
    rank_indices_surrogate = np.argsort(np.argsort(x_shuffled))
    x_surrogate[rank_indices_surrogate] = sorted_x

    return x_surrogate


# Generate surrogate
aaft_x = aaft_surrogate(x_orig)

# Plot original and surrogate time series
plt.figure(figsize=(12, 5))
plt.subplot(2, 1, 1)
plt.plot(t, x_orig, label='Original')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, aaft_x, label='AAFT Surrogate', color='r')
plt.legend()
plt.show()

# Compare power spectra
f_orig = np.abs(fft(x_orig))
f_aaft = np.abs(fft(aaft_x))

plt.figure(figsize=(12, 5))
plt.plot(f_orig[:len(f_orig) // 2], label='Original Power Spectrum')
plt.plot(f_aaft[:len(f_aaft) // 2], label='AAFT Power Spectrum', linestyle='dashed')
plt.legend()
plt.show()

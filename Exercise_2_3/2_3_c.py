import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm

# Function to generate the Rössler system
def rossler(x, y, z, a=0.2, b=0.2, c=5.0):
    dxdt = -y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    return np.array([dxdt, dydt, dzdt])

# 4th-order Runge-Kutta solver for the Rössler system
def RK4(system, state, dt, params):
    k1 = dt * system(*state, *params)
    k2 = dt * system(*(state + 0.5 * k1), *params)
    k3 = dt * system(*(state + 0.5 * k2), *params)
    k4 = dt * system(*(state + k3), *params)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Simulate the Rössler system
def simulate_rossler(initial_state, t_end, dt, params):
    steps = int(t_end / dt)
    trajectory = np.zeros((steps, 3))
    trajectory[0] = initial_state
    for i in range(steps - 1):
        trajectory[i + 1] = RK4(rossler, trajectory[i], dt, params)
    return trajectory

# Extract local maxima of x(t)
def extract_local_maxima(data):
    #peaks, _ = find_peaks(data)
    peaks, _ = find_peaks(data, height=0.5, distance=50, prominence=0.1)
    return data[peaks]

# Task (b): Plot xmax vs. xmax[i+1]
def plot_map(x_maxima):
    plt.figure(figsize=(8, 6))
    plt.plot(x_maxima[:-1], x_maxima[1:], 'o', markersize=3)
    plt.title("One-dimensional Map (xmax[i] vs. xmax[i+1])")
    plt.xlabel("$x_{max}[i]$")
    plt.ylabel("$x_{max}[i+1]$")
    plt.grid()
    plt.show()

# Task (c): Bifurcation diagram for 2 < c < 6
def bifurcation_diagram(initial_state, t_end, dt, c_range, a=0.2, b=0.2):
    plt.figure(figsize=(10, 8))
    for c in tqdm(np.linspace(*c_range, 500)):
        # Simulate the Rössler system
        params = (a, b, c)
        trajectory = simulate_rossler(initial_state, t_end, dt, params)
        x = trajectory[:, 0]

        # Extract local maxima of x(t)
        x_maxima = extract_local_maxima(x)

        # Plot the bifurcation points
        plt.plot([c] * len(x_maxima), x_maxima, ',k', markersize=1)

    plt.title("Bifurcation Diagram")
    plt.xlabel("$c$")
    plt.ylabel("$x_{max}$")
    plt.grid()
    plt.show()

# Parameters
initial_state = np.array([1.0, 1.0, 1.0])  # Initial conditions
t_end = 1000  # Total simulation time
dt = 0.01  # Time step

# Task (b)
params_b = (0.2, 0.2, 5.0)  # a, b, c for c=5
trajectory_b = simulate_rossler(initial_state, t_end, dt, params_b)
x_maxima = extract_local_maxima(trajectory_b[:, 0])
print(x_maxima)
plot_map(x_maxima)

# Task (c)
c_range = (2.0, 6.0)  # Range of c for bifurcation diagram
t_end_c = 500  # Shorter simulation time for bifurcation diagram
bifurcation_diagram(initial_state, t_end_c, dt, c_range)


# Find local maxima


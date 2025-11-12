import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# set parameters for lorenz system
a = 0.2
b = 0.2
c = 5
steps = 100000
dt = 1e-2

# define Lorenz system
def Rössler(x,y,z):
    dxdt = -y -z
    dydt = x + a*y
    dzdt = b + z*(x - c)
    return np.array([dxdt, dydt, dzdt])

# define 4th order Runge-Kutta method
def RK4(system, state, dt):
    k1 = dt*system(*state)
    k2 = dt * system(*(state + 0.5 * k1))
    k3 = dt * system(*(state + 0.5 * k2))
    k4 = dt * system(*(state + k3))
    return state + (k1 + 2*k2 + 2*k3 + k4)/6

# Solve a given system with RK4
def solve_system(system, initial_state, steps, dt):
    trajectory = np.zeros((steps + 1, 3))
    trajectory[0] = initial_state

    for i in tqdm(range(steps)):
        trajectory[i+1] = RK4(system,trajectory[i],dt)

    discard = 3*len(trajectory)//4
    return trajectory[discard:, 0], trajectory[discard:, 1], trajectory[discard:, 2]

# solve Lorenz system with initial conditions
initial_state = np.array([1.0,1.0,1.0])
x_val, y_val, z_val = solve_system(Rössler, initial_state, steps, dt)

# Plot the trajectory
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_val, y_val, z_val, lw=0.5)
ax.scatter(x_val[0], y_val[0], z_val[0], color='red', s=10)
ax.set_title("Rössler Attractor")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()


from scipy.signal import find_peaks

# Example time series
time = np.arange(steps/4)
data = x_val[:-1]

# Find local maxima
peaks, _ = find_peaks(data, height=0, prominence=0.05)  # Adjusted parameters

# Plot the time series and mark the peaks
plt.figure(figsize=(10, 6))
plt.plot(time, data, label="Time Series")
plt.plot(time[peaks], data[peaks], "ro", label="Local Maxima")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Local Maxima in Time Series")
plt.legend()
plt.grid()
plt.show()

maxima = data[peaks]
plt.scatter(maxima[:-1], maxima[1:])
plt.show()
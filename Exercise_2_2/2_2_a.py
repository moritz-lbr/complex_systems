import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# set parameters for lorenz system
sigma = 10
rho = 28
beta = 8/3
steps = 10000
dt = 1e-2

# define Lorenz system
def lorenz(x,y,z):
    dxdt = sigma*(y - x)
    dydt = x*(rho-z) - y
    dzdt = x*y - beta*z
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

    return trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]

# Create 3D plot figure
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

for i in range(2):
    # solve Lorenz system with initial conditions
    initial_state = np.array(np.random.uniform(-20,20, size=(1,3))[0])
    x_val, y_val, z_val = solve_system(lorenz, initial_state, steps, dt)

    # Plot the trajectories
    ax.plot(x_val, y_val, z_val, lw=0.5)
    ax.scatter(*initial_state, s=10)
    ax.set_title("Lorenz Attractor")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

plt.show()
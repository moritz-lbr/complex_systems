import numpy as np
import matplotlib.pyplot as plt

# Set parameters for Lorenz system
sigma = 10
rho = 28
beta = 8 / 3
steps = 10000
dt = 1e-2

# Define the Lorenz system
def lorenz(x, y, z):
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])

# Define 4th order Runge-Kutta method
def RK4(system, state, dt):
    k1 = dt * system(*state)
    k2 = dt * system(*(state + 0.5 * k1))
    k3 = dt * system(*(state + 0.5 * k2))
    k4 = dt * system(*(state + k3))
    return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Solve the Lorenz system
def solve_system(system, initial_state, steps, dt):
    trajectory = np.zeros((steps + 1, 3))  # Array to hold x, y, z
    trajectory[0] = initial_state  # Set initial conditions

    for i in range(steps):
        trajectory[i + 1] = RK4(system, trajectory[i], dt)

    return trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]

# Solve up to t = 25
t_end = 25
steps_t_end = int(t_end / dt)
initial_state = np.array([1.0, 1.0, 1.0])
x, y, z = solve_system(lorenz, initial_state, steps_t_end, dt)

# Extract final state at t = 25
state_t25 = np.array([x[-1], y[-1], z[-1]])

# Perturb the initial state slightly
perturbation = 1e-4
state_perturbed = state_t25 + np.array([perturbation, 0, 0])

# Solve for t = 25 to t = 50
t_start = 25
t_end = 50
steps_t50 = int((t_end - t_start) / dt)
t_values = np.linspace(t_start, t_end, steps_t50 + 1)

x1, _, _ = solve_system(lorenz, state_t25, steps_t50, dt)         # Unperturbed
x2, _, _ = solve_system(lorenz, state_perturbed, steps_t50, dt)  # Perturbed

# Plot x(t) for both cases
plt.figure(figsize=(10, 6))
plt.plot(t_values, x1, label="Unperturbed (initial state at t=25)")
plt.plot(t_values, x2, label="Perturbed (initial state + 1e-4)")
plt.xlabel("Time (t)")
plt.ylabel("x(t)")
plt.title("Sensitivity to Initial Conditions in the Lorenz System")
plt.legend()
plt.grid()
plt.show()

# Logarithmic divergence growth
plt.figure(figsize=(10, 6))
plt.plot(t_values, np.abs(x1 - x2), label="|x1(t) - x2(t)|")
plt.yscale("log")
plt.xlabel("Time (t)")
plt.ylabel("Logarithmic Error |x1(t) - x2(t)|")
plt.title("Error Growth Between Trajectories")
plt.legend()
plt.grid()
plt.show()

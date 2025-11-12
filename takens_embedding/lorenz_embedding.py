import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.neighbors import NearestNeighbors

# Lorenz system parameters
sigma, rho, beta = 10, 28, 8/3

# Define the Lorenz system
def lorenz(t, state):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# Solve the Lorenz system
t_max = 50
dt = 0.01
t = np.arange(0, t_max, dt)
initial_state = [1.0, 1.0, 1.0]
sol = solve_ivp(lorenz, [0, t_max], initial_state, t_eval=t)

# Extract the x-component (observed variable)
x_t = sol.y[0]

# Delay embedding parameters
tau = 10  # Delay (in units of dt)
m = 3  # Embedding dimension

# Create the delay-embedded space
N = len(x_t) - (m - 1) * tau  # Number of valid points
X_embedded = np.array([x_t[i:N + i] for i in range(0, m * tau, tau)]).T
print(X_embedded)
# Plot the original Lorenz attractor
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(sol.y[0], sol.y[1], sol.y[2], lw=0.5, color='blue')
ax1.set_title("Original Lorenz Attractor")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

# Plot the reconstructed attractor from delay embedding
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], lw=0.5, color='red')
ax2.set_title("Reconstructed Attractor via Takens' Embedding")
ax2.set_xlabel("s(t)")
ax2.set_ylabel("s(t-τ)")
ax2.set_zlabel("s(t-2τ)")

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy.lib.scimath as scimath

# Iterations
iterations = 1000

# Parameters
stepsize = 0.002
b_range = np.arange(-0.98, 1, stepsize)
x_start = 1
y_start = 1

def x0(a, b):
    return (np.sqrt(4 * a - 3 * (b - 1) ** 2) + b - 1) / (2 * a)

def x1(a, b):
    return (-np.sqrt(4 * a - 3 * (b - 1) ** 2) + b - 1) / (2 * a)

def lamb(a,b,x1,x2):
    trace = 4 * (a ** 2) * x1 * x2 + 2 * b
    det = b ** 2
    lamb_plus = trace/2 + 1/2 * scimath.sqrt(trace**2 - 4 * det)
    lamb_minus = trace/2 - 1/2 * scimath.sqrt(trace**2 - 4 * det)
    if np.abs(lamb_plus) < 1 and np.abs(lamb_minus) < 1:
        return True
    else:
        return False

# Initial guess for two-cycles x and y
initial_guess = np.array([1, 1])

# Boundary values a1
a1_vals = []
b_a1_vals = []

# Stable and unstable two-cycles
a_stable = []
b_stable = []
a_unstable = []
b_unstable = []

# Iteration for all a and b values:
for j in tqdm(b_range):
    a_limit = -0.5
    a_values = np.arange(start=a_limit, stop=4.5, step=stepsize)
    a1 = 3 / 4 * (1 - j) ** 2
    a1_vals.append(a1)
    b_a1_vals.append(j)

    for i in a_values:
        if i > a1:

            if lamb(a=i, b=j, x1=x0(a=i, b=j), x2=x1(a=i, b=j)):
                a_stable.extend([i])
                b_stable.extend([j])
            else:
                a_unstable.extend([i])
                b_unstable.extend([j])
        else:
            continue

# Plotting
fig1, ax1 = plt.subplots()
ax1.scatter(b_stable, a_stable, color='green', marker='o', s=1, label=r'$|det[J(x_1) \cdot J(x_2)]| < 1$ Stable Two-cycle')
ax1.scatter(b_unstable, a_unstable, color='red', marker='o', s=1, label=r'$|det[J(x_1) \cdot J(x_2))]| > 1$ Unstable Two-cycle')
ax1.scatter(b_a1_vals, a1_vals, color='blue', marker='o', s=1, label=r'$a_1(b) = \frac{3}{4}(1-b)^2$')
ax1.set_xlabel(r'[b]', fontsize=25)
ax1.set_ylabel(r'[a]', fontsize=25)
ax1.set_title(r'Stability analysis of the 2-Cycles for possible parameter pairs (a,b)', fontsize=25)
plt.legend(loc='lower left', bbox_to_anchor=(0, 0), fontsize=18, markerscale=10)
textstr = (f"Simulation Parameters:\n"
           f"Stepsize (a,b): {stepsize}"
           f"\n"
           r"$a(b) > a_1(b)$"
           f"\n"
           r"$|b| < 1$")
plt.gcf().text(0.14, 0.43, textstr, fontsize=18, va='top',
               bbox=dict(facecolor='white', alpha=1))
ax1.set_ylim(-1,4.5)
plt.show()

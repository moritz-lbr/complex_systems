import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy.lib.scimath as scimath

np.seterr(over='raise')

# Iterations
iterations = 1000
accumulation_points = 100

# Parameters
stepsize = 0.1
b_range = [-0.45, -0.25, -0.05, 0.15, 0.35, 0.55]
x_start = 1
y_start = 1

def henon(iter, a, b, x0, y0):
    x_val = [x0]
    y_val = [y0]

    for i in range(iter):
        x = y_val[-1] + 1 - a * (x_val[-1]) ** 2
        y = b * x_val[-1]
        x_val.append(x)
        y_val.append(y)

    x_accum_points = x_val[-accumulation_points:]
    y_accum_points = y_val[-accumulation_points:]
    return x_accum_points, y_accum_points

def lamb(a,b,x1,x2):
    trace = 4 * (a ** 2) * x1 * x2 + 2 * b
    det = b ** 2
    lamb_plus = trace/2 + 1/2 * scimath.sqrt(trace**2 - 4 * det)
    lamb_minus = trace/2 - 1/2 * scimath.sqrt(trace**2 - 4 * det)
    if np.abs(lamb_plus) < 1 and np.abs(lamb_minus) < 1:
        return True
    else:
        return False
def x0(a, b):
    return (np.sqrt(4 * a - 3 * (b - 1) ** 2) + b - 1) / (2 * a)

def x1(a, b):
    return (-np.sqrt(4 * a - 3 * (b - 1) ** 2) + b - 1) / (2 * a)


# Initial guess for two-cycles x and y
initial_guess = np.array([1, 1])

# Fixpoints x,y with a-values
fixpoints_x = []
fixpoints_y = []
fixpoints_a_vals = []
fixpoints_b_vals = []

# Boundary values a1
a1_vals = []
b_a1_vals = []
a1_vals_x = []
a1_vals_y = []

# Stable and unstable two-cycles
x_stable = []
x_unstable = []
a_stable = []
b_stable = []
a_unstable = []
b_unstable = []

# Iteration for all a and b values:
for j in tqdm(b_range):
    a_limit = -0.5
    a_values = np.arange(start=a_limit, stop=2.5, step=0.001)

    a1 = 3 / 4 * (1 - j) ** 2
    a1_vals.append(a1)
    b_a1_vals.append(j)
    a_limit_x, a_limit_y = henon(iter=iterations, a=a1, b=j, x0=x_start, y0=y_start)
    a1_vals_x.append(a_limit_x[0]), a1_vals_y.append(a_limit_y[0])

    for i in a_values:
        try:
            points_x, points_y = henon(iter=iterations, a=i, b=j, x0=x_start, y0=y_start)
            if i > a1:
                if lamb(a=i, b=j, x1=x0(a=i, b=j), x2=x1(a=i, b=j)):
                    x_stable.extend(points_x)
                    a_stable.extend([i] * accumulation_points)
                    b_stable.extend([j] * accumulation_points)
                else:
                    x_unstable.extend(points_x)
                    a_unstable.extend([i] * accumulation_points)
                    b_unstable.extend([j] * accumulation_points)

            else:
                fixpoints_x.extend(points_x)
                fixpoints_y.extend(points_y)
                fixpoints_a_vals.extend([i] * accumulation_points)
                fixpoints_b_vals.extend([j] * accumulation_points)

        except FloatingPointError:
            # print(f"Overflow encountered at a-value: {np.round(i, 3)}. \n"
            #       f"Terminating iteration process for this parameter pair (a={np.round(i, 3)},b={j})"
            #       f"and continue with next iteration pair.\n")
            continue

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for different points
for i in range(len(a1_vals)):
    a1_sc = ax.scatter(a1_vals[i], b_range[i], a1_vals_x[i], c='red', marker='o', facecolors='none', s=30, label=r'$a_1 = \frac{3}{4}(1-b)^2$')
sc = ax.scatter(fixpoints_a_vals, fixpoints_b_vals, fixpoints_x, c='#4682B4', marker='o', s=0.001, label='Fix Points')
a_stable_sc = ax.scatter(a_stable, b_stable, x_stable, c='blue', marker='o', s=0.001, label='Stable Two-Cycle')
a_unstable_sc = ax.scatter(a_unstable, b_unstable, x_unstable, c='purple', marker='o', s=0.001, label='Higher Order Cycles')

# Set axis limits
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(-4, 4)

# Labels and title
ax.set_xlabel('[a]', fontsize=20)
ax.set_ylabel('[b]', fontsize=20)
ax.set_zlabel('[X-Coordinate]', fontsize=20)
ax.set_title('3D Scatter Plot of the accumulation points of the X-coordinate (a,b) of the Henon Map', fontsize=20)

# Adjust 3D view
elevation = 40
azimuth = 60
ax.view_init(elev=elevation, azim=azimuth)

# First legend (Group of points for fix points and two cycles)
legend1 = ax.legend(handles=[sc, a_stable_sc, a_unstable_sc], markerscale=300, loc='upper right', bbox_to_anchor=(1.55, 1), fontsize=20)
ax.add_artist(legend1)

# Second legend for a1 scatter points
legend2 = ax.legend(handles=[a1_sc], markerscale=2, loc='upper right', bbox_to_anchor=(1.42, 0.8), fontsize=20)

# Show plot
plt.show()

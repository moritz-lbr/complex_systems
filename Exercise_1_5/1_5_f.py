import matplotlib.pyplot as plt
import numpy as np
import tqdm
np.seterr(over='raise')

# Iterations
iterations = 1000
accumulation_points = 400

# Parameters
stepsize = 0.001
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


acc_points_x = []
acc_points_y = []
a_vals = []
b_vals = []

b = 0.3
a_values = np.arange(start=-1, stop=1.5, step=stepsize)

for i in tqdm.tqdm(a_values):
    try:
        points_x, points_y = henon(iter=iterations, a=i, b=b, x0=x_start, y0=y_start)
        acc_points_x.extend(points_x)
        acc_points_y.extend(points_y)
        a_vals.extend([i]*accumulation_points)
    except FloatingPointError:
        print(f"Overflow encountered at a-value: {np.round(i,3)}. \n"
              f"Terminating iteration process for this parameter pair (a={np.round(i,3)},b={b})"
              f"and continue with next iteration pair.\n")
        continue


fig, ax = plt.subplots()
ax.vlines(1.06, linestyles='--', color='red', label=r'$a = 1.06$', linewidth=1, ymin=-1,ymax=1.5)
ax.scatter(a_vals,acc_points_x, marker='o', s=0.005, color='black', label='Accumulation points of $x_{n} (a,b=0.3)$')
ax.set_title(r'Bifurcation diagram of the X-Coordinate of the Henon map', fontsize=20)
ax.set_xlabel(r'$[a]$', fontsize=20)
ax.set_ylabel(r'[X-Coordinate]', fontsize=20)

textstr = (f"Simulation Parameters:\n"
           f"Stepsize (for a): {stepsize}\n"
           r"$ -1 < a < 1.5$"
           f"\n"
           r"$b = 0.3$"
           f"\n"
           rf"$X_0 = {x_start}$"
           f"\n"
           rf"$Y_0 = {y_start}$")
plt.gcf().text(0.137, 0.25, textstr, fontsize=15, bbox=dict(facecolor='white', alpha=1))
plt.legend(loc='lower left', markerscale=100, fontsize=20)

fig2, ax2 = plt.subplots()
ax2.vlines(1.06, linestyles='--', color='red', label=r'$a = 1.06$', linewidth=1, ymin=-1,ymax=1.5)
ax2.scatter(a_vals,acc_points_x, marker='o', s=0.005, color='black', label='Accumulation points of $x_{n} (a,b=0.3)$')
ax2.set_title(r'Bifurcation diagram of the X-Coordinate of the Henon map', fontsize=20)
ax2.set_xlabel(r'$[a]$', fontsize=20)
ax2.set_ylabel(r'[X-Coordinate]', fontsize=20)
ax2.set_xlim(0.9, 1.4)

plt.show()

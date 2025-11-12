import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import Counter
np.seterr(over='raise')

# Iterations
iterations = 1000

# Parameters
x_start = 0.95
y_start = 0.95

def henon(iter, a, b, x0, y0):
    x_val = [x0]
    y_val = [y0]

    for i in tqdm(range(iter)):
        x = y_val[-1] + 1 - a * (x_val[-1]) ** 2
        y = b * x_val[-1]
        x_val.append(x)
        y_val.append(y)

    x_accum_points = x_val[5:]
    y_accum_points = y_val[5:]
    return x_accum_points, y_accum_points


acc_points_x = []
acc_points_y = []
a_vals = []
b_vals = []

b = 0.3
a = 1.3

points_x, points_y = henon(iter=iterations, a=a, b=b, x0=x_start, y0=y_start)
acc_points_x.extend(points_x)
acc_points_y.extend(points_y)

xy_pairs = list(zip(acc_points_x, acc_points_y))
counts = Counter(xy_pairs)

# Map each point to its count
colors = [counts[(xi, yi)] for xi, yi in xy_pairs]


plt.scatter(acc_points_x,acc_points_y, c=colors,cmap='viridis', marker='o', s=5)
#plt.colorbar(label='Point Density')  # Add colorbar
plt.colorbar(label='Accumulation points of $x_{n} (a,b=0.3)$')
plt.title(rf'Henon attractor (a={a}, b={b}) with starting conditions ($X_0={x_start}$, $Y_0={y_start}$)', fontsize=20)
plt.xlabel(r'$[X_n]$', fontsize=20)
plt.ylabel(r'$[Y_n]$', fontsize=20)
plt.show()

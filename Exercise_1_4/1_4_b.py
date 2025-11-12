import matplotlib.pyplot as plt
import numpy as np
import tqdm

# Parameters
r_spacing = 0.001
r = np.arange(0, 4.0, r_spacing)
x0 = 0.1

# Iterations
iterations = 1000
accumulation_points = 400

def logistic(iter, r_val, x_start):
    x = x_start
    values = []

    for i in range(iter):
        values.append(x)
        x = x*r_val*(1.0-x)

    accum_points = values[-accumulation_points:]

    return accum_points

acc_points = []
r_values = []

for i in tqdm.tqdm(r):
    points = logistic(iter=iterations, r_val=i, x_start=x0)
    acc_points.extend(points)
    r_values.extend([i]*accumulation_points)


fig, ax = plt.subplots()
ax.scatter(r_values,acc_points, marker='o', s=0.005, color='black', label='$x_{n}$ after n iterations')
#ax.hlines(x_limit_value, colors='red', xmin=0, xmax=xlim, linestyles='--', label=rf'Analytical limit value $x*=1-1/r = {x_limit_value:.4f}$')
ax.set_title(r'Bifurcation diagram of the logistic map $x_{n+1} = rx_{n}(1-x_{n})$', fontsize=20)
ax.set_xlabel('$r$-value', fontsize=20)
ax.set_ylabel('Accumulation points of $x_n$', fontsize=20)
# Create a text box with parameters
textstr = (f"Simulation Parameters:\nIterations/$r$-value = {iterations} \n$x_{'0'}$ = {x0}"
           f"\n$r$-values: [0,4] in steps of {r_spacing} \n\nExplanation:\n"
           f"The last {accumulation_points} values of "
           f"$x_{'n'}$ obtained from the respective \nsimulation,"
           f"characterized by the corresponding $r$-value,\nare shown"
           f"in this bifurcation diagram.")
# Place text box in the plot
plt.gcf().text(0.15, 0.6, textstr, fontsize=15, bbox=dict(facecolor='white', alpha=1))

plt.show()

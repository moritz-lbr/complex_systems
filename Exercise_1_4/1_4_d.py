import matplotlib.pyplot as plt
import numpy as np

# Parameters
r = 3.1
x0 = 0.1

# Iterations
iterations = 1000

def logistic(iter, r_val, x_start):
    x = x_start
    x_limit = 1.0 - 1.0/r_val
    values = []

    for i in range(iter):
        values.append(x)
        x = x*r_val*(1.0-x)

    deviation = x_limit - values[-1]

    return x_limit, deviation, values



x_limit_value, delta, x_sequence = logistic(iter=iterations,
                                            r_val=r,
                                            x_start=x0)

print(f'\nSimulation parameters: \nx_0 = {x0} \nr = {r} \niterations: {iterations} \n\n'
      f'Analytical limit value: {x_limit_value} \n\n'
      f'First 10 iterations of logistic map sequence with listed setup parameters: {x_sequence[:10]} \n\n'
      f'Last entry of logistic map sequence after {iterations} iterations: {x_sequence[-1]} \n\n'
      f'Deviation of last entry from logistic map sequence and analytical limit value: {delta}')


x_map = np.arange(0, 1.0, 0.01)
y_map = [x*r*(1.0-x) for x in np.arange(0, 1.0, 0.01)]

x_web = []
y_web = []

for i in range(len(x_sequence[:50])):
    x_web.extend([x_sequence[i]]*2)
    y_web.append(x_sequence[i])
    y_web.append(x_sequence[i+1])


fig, ax = plt.subplots()
ax.plot([0,1],[0,1], color='blue', label=f'$y = x$')
ax.plot(x_map,y_map, color='black', label=f'$y = {r}x(1-x)$')
ax.plot(x_web,y_web, marker='o', color='red', label='$x_{n+1}$ after n iterations')
ax.set_title(r'Cobweb diagram of the logistic map', fontsize=20)
ax.set_xlabel('$x$', fontsize=20)
ax.set_ylabel('$y$', fontsize=20)
plt.legend(loc='lower right', fontsize=20)

textstr = (f"Simulation Parameters:\nIterations = {iterations} \n$x_{'0'}$ = {x0}"
           f"\n$r$-value = {r} \n\nExplanation:\n"
           f"Cobweb diagram for the logistic map: \n"
           f"The depicted points alternately show the evaluation\n"
           f"of the two functions $y = {r}x(1-x)$ and $y = x$.")
# Place text box in the plot
plt.gcf().text(0.15, 0.67, textstr, fontsize=12, bbox=dict(facecolor='white', alpha=1))

plt.show()

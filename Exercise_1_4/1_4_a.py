import matplotlib.pyplot as plt

# Parameters
r = 1.0/0.8
x0 = 0.5

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

xlim = 100

fig, ax = plt.subplots()
ax.plot(range(xlim),x_sequence[:xlim], marker='o', label=rf'$x_n$ after n iterations, $x_0 = {x0}$')
ax.hlines(x_limit_value, colors='red', xmin=0, xmax=xlim, linestyles='--', label=rf'Analytical limit value $x*(r={r:.2f})= {x_limit_value:.2f}$')
ax.set_title(r'Convergence of logistic mapping to analytical limit value $x*=1-1/r$', fontsize=20)
ax.set_xlabel('Iteration (n)', fontsize=20)
ax.set_ylabel('$x_n$', fontsize=20)
plt.legend(loc='center right', fontsize=20)
plt.show()

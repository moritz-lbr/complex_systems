import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def lyapunov(f, fprime, vars, x0, n):
    lamb: int = 0
    xn = f(*vars, x0, n)
    for i in range(n):
        lamb += np.log(np.abs(fprime(*vars, xn[i])))
    return lamb/n, n

def logistic(r_val, x0, iter):
    x = x0
    values = []
    for i in range(iter):
        values.append(x)
        x = x*r_val*(1.0-x)
    return np.array(values)

def logistic_prime(r_val, xi):
    return r_val*(1 - 2*xi)


# calculate lyapunov exponent for logistic map:
r_spacing = 0.001
r_vals = np.arange(0,4,r_spacing)
N = 10000
x0 = 0.1

lamb_r = [lyapunov(logistic, logistic_prime, [r],x0,N)[0] for r in tqdm(r_vals)]


fig,ax = plt.subplots()
for i, lamb in enumerate(lamb_r):
    if lamb < 0:
        ax.scatter(r_vals[i],lamb, marker='o', color='black', s=1)
    else:
        ax.scatter(r_vals[i],lamb, marker='o', color='red', s=1)

ax.scatter([],[], color='black', label=r'Non-Chaotic Behaviour: $\lambda_r < 0$')
ax.scatter([],[], color='red', label=r'Chaotic Behaviour: $\lambda_r > 0$')

ax.set_title(r'Lyapunov exponents of the logistic map', fontsize=20)
ax.set_xlabel('$r$-value', fontsize=20)
ax.set_ylabel(r'$\lambda_{r}$', fontsize=20)
# Create a text box with parameters
textstr = (f"Simulation Parameters:\nIterations/$r$-value = {N} \n$x_{'0'}$ = {x0}"
           f"\n$r$-values: [0,4] in steps of {r_spacing}\n"
           r"$\lambda_{r} = \lim_{n \to \infty} \frac{1}{n} \sum_{i=1}^{n} \ln \left| f'_{r}(x_i) \right|$"
           f"\n"
           r"$f'_{r}(x_{i}) = r(1-2x_{i})$")

# Place text box in the plot
plt.gcf().text(0.65, 0.23, textstr, fontsize=15, bbox=dict(facecolor='white', alpha=1))
plt.legend(loc='lower right', markerscale=1, fontsize=15)
plt.show()

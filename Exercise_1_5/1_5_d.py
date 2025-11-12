import numpy as np
import numpy.lib.scimath as scimath
import matplotlib.pyplot as plt
from tqdm import tqdm

stepsize = 0.005
def fixed_points(a,b):
    calc1 = -(1 - b)
    calc2 = np.sqrt((1 - b) ** 2 + 4 * a)
    x_plus_val = (calc1 + calc2) / (2 * a)
    x_minus_val = (calc1 - calc2) / (2 * a)
    return x_plus_val, x_minus_val

def lamb(a,b,x):
    lamb_plus = -a * x + scimath.sqrt((a * x) ** 2 + b)
    lamb_minus = -a * x - scimath.sqrt((a * x) ** 2 + b)
    if np.abs(lamb_plus) < 1 and np.abs(lamb_minus) < 1:
        return True
    else:
        return False

b_values = np.arange(start=-0.9999, stop=0.9999, step=stepsize)

stable_x_plus_a = []
stable_x_plus_b = []

stable_x_minus_a = []
stable_x_minus_b = []

unstable_x_plus_a = []
unstable_x_plus_b = []

unstable_x_minus_a = []
unstable_x_minus_b = []

x_complex_a = []
x_complex_b = []

a1_x_plus = []

for b in tqdm(b_values):
    a_limit = -((1 - b) ** 2) / 4
    a_values = np.arange(start=a_limit + 1e-4, stop=4, step=stepsize)
    a_forbidden = np.arange(start=-1,stop=a_limit - 1e-4, step=stepsize)

    a1_x_plus.append((3 / 4) * (1 - b) ** 2)

    for a_forb in a_forbidden:
        x_complex_a.append(a_forb)
        x_complex_b.append(b)

    for a in a_values:
        x_plus, x_minus = fixed_points(a,b)

        if lamb(a,b,x_plus):
            stable_x_plus_a.append(a)
            stable_x_plus_b.append(b)
        else:
            unstable_x_plus_a.append(a)
            unstable_x_plus_b.append(b)

        if lamb(a,b,x_minus):
            stable_x_minus_a.append(a)
            stable_x_minus_b.append(b)
        else:
            unstable_x_minus_a.append(a)
            unstable_x_minus_b.append(b)


fig1, ax1 = plt.subplots()
ax1.scatter(stable_x_plus_b, stable_x_plus_a, color='green', marker='o', s=5, label=r'$|\lambda_{\pm}[x_{+}(a,b)]| < 1$ (stable)')
ax1.scatter(unstable_x_plus_b, unstable_x_plus_a, color='red', marker='o', s=5, label=r'$|\lambda_{\pm}[x_{+}(a,b)]| > 1$ (unstable)')
ax1.scatter(x_complex_b, x_complex_a, color='black', marker='o', s=5, label=r'$x_{+}(a,b) \in \mathbb{C}$ (complex)')
ax1.scatter(b_values, a1_x_plus, color='blue', marker='o', s=5, label=r'$\lambda_{-}[x_{+}(a_1,b)] = -1$')
ax1.set_xlabel(r'[b]', fontsize=25)
ax1.set_ylabel(r'[a]', fontsize=25)
ax1.set_title(r'Stability analysis of the fixed point $x_{+}$ for possible parameter pairs (a,b)', fontsize=25)
plt.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95), fontsize=20, markerscale=5)
textstr = (f"Simulation Parameters:\n"
           f"Stepsize (for a,b): {stepsize}\n"
           r"$a > a_0 = - \frac{(1-b)^2}{4}$"
           f"\n"
           r"$a_1 = \frac{3}{4}(1-b)^2$"
           f"\n"
           r"$|b| < 1$")
plt.gcf().text(0.62, 0.6, textstr, fontsize=20, va='top',
               bbox=dict(facecolor='white', alpha=1))

fig2, ax2 = plt.subplots()
ax2.scatter(stable_x_minus_b, stable_x_minus_a, color='green', marker='o', s=5, label=r'$|\lambda_{\pm}[x_{-}(a,b)]| < 1$ (stable)')
ax2.scatter(unstable_x_minus_b, unstable_x_minus_a, color='red', marker='o', s=5, label=r'$|\lambda_{\pm}[x_{-}(a,b)]| > 1$ (unstable)')
ax2.scatter(x_complex_b, x_complex_a, color='black', marker='o', s=5, label=r'$x_{-}(a,b) \in C$ (complex)')
ax2.set_xlabel(r'[b]', fontsize=25)
ax2.set_ylabel(r'[a]', fontsize=25)
ax2.set_title(r'Stability analysis of the fixed point $x_{-}$ for possible parameter pairs (a,b)', fontsize=25)
plt.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95), fontsize=20, markerscale=5)
textstr = (f"Simulation Parameters:\n"
           f"Stepsize (for a,b): {stepsize}\n"
           r"$a > a_0 = - \frac{(1-b)^2}{4}$"
           f"\n"
           r"$|b| < 1$")
plt.gcf().text(0.62, 0.65, textstr, fontsize=20, va='top',
               bbox=dict(facecolor='white', alpha=1))
plt.show()



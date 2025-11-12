import matplotlib.pyplot as plt
import numpy as np

# Parameters
a = 1.4
b = 0.3
x_start = 1
y_start = 2

# Iterations
iterations = 5

def henon(iter, a, b, x0, y0):
    x = x0
    y = y0
    x_val = [x0]
    y_val = [y0]

    for i in range(iter):
        x = y_val[-1] + 1 - a * (x_val[-1])**2
        y = b * x_val[-1]
        x_val.append(x)
        y_val.append(y)

    x_inv_val = [x]
    y_inv_val = [y]

    for i in range(iter):
        x = y_inv_val[-1]/b
        y = x_inv_val[-1] - 1 + a/(b**2) * (y_inv_val[-1])**2
        x_inv_val.append(x)
        y_inv_val.append(y)

    x_val = np.round(np.array(x_val),3)
    y_val = np.round(np.array(y_val),3)
    x_inv_val = np.round(np.array(x_inv_val), 3)
    y_inv_val = np.round(np.array(y_inv_val), 3)

    return x_val, y_val, x_inv_val, y_inv_val

x,y,x_inv, y_inv = henon(iterations, a=a, b=b, x0=x_start, y0=y_start)

np.set_printoptions(suppress=True, precision=3)

print(f'x-value sequence obtained from Henon Map (x0 = {x_start}): {x}')
print(f'Inverted x-value sequence obtained from inverted Henon Map (x0 = {x[-1]}): {x_inv}')
print(f'y-value sequence obtained from Henon Map (y0 = {y_start}): {y}')
print(f'Inverted y-value sequence obtained from inverted Henon Map (y0 = {y[-1]}): {y_inv}')
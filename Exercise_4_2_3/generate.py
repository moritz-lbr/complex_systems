import numpy as np

delta_t = 10 ** -3
sigma, rho, beta = 10, 28, 8 / 3  # lorenz
a = 1.3  # halvorson

t_disc = 50


def halvorsen(t, state):
    x, y, z = state
    dxdt = -a * x - 4 * y - 4 * z - y ** 2
    dydt = -a * y - 4 * z - 4 * x - z ** 2
    dzdt = -a * z - 4 * x - 4 * y - x ** 2
    return np.array([dxdt, dydt, dzdt])


def lorenz(t, state):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])


def state_n(f, t, state):
    k1 = f(t, state)
    k2 = f(t + delta_t / 2, state + delta_t * k1 / 2)
    k3 = f(t + delta_t / 2, state + delta_t * k2 / 2)
    k4 = f(t + delta_t, state + delta_t * k3)
    return state + delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def generate_lorenz(n_ges):
    x0, y0, z0 = 0.2, 0.6, -0.3
    t = np.arange(0, n_ges * delta_t + t_disc, delta_t)
    states = np.zeros((len(t), 3))
    t_1 = np.where(t == t_disc)[0][0]

    state0 = np.array([x0, y0, z0])
    # iteration RK4
    for i in range(len(t)):
        next_state = state_n(lorenz, t[i], state0)
        states[i, :] = next_state
        state0 = next_state
    return states, states[t_1:]


def generate_halvorsen(n_ges):
    x0, y0, z0 = 2, 1, 1
    t = np.arange(0, n_ges * delta_t + t_disc, delta_t)
    states = np.zeros((len(t), 3))
    t_1 = np.where(t == t_disc)[0][0]

    state0 = np.array([x0, y0, z0])
    # iteration RK4
    for i in range(len(t)):
        next_state = state_n(halvorsen, t[i], state0)
        states[i, :] = next_state
        state0 = next_state
    return states, states[t_1:]

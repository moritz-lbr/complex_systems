import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import scipy.sparse.linalg
from sklearn.linear_model import Ridge


# Create Timeseries for Systems:
# define Lorenz system
def lorenz(x, y, z, sigma=10, rho=28, beta=8/3):
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])

# define halverson system
def halverson(x, y, z, sigma=1.3):
    dxdt = -sigma*x -4*y -4*z -y**2
    dydt = -sigma*y -4*z -4*x -z**2
    dzdt = -sigma*z -4*x -4*y -x**2
    return np.array([dxdt, dydt, dzdt])

# define 4th order Runge-Kutta method
def RK4(system, state, dt):
    k1 = dt * system(*state)
    k2 = dt * system(*(state + 0.5 * k1))
    k3 = dt * system(*(state + 0.5 * k2))
    k4 = dt * system(*(state + k3))
    return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# solve a given system with RK4
def solve_system(system, initial_state, steps, dt):
    trajectory = np.zeros((steps + 1, 3))
    trajectory[0] = initial_state

    for i in tqdm(range(steps)):
        trajectory[i + 1] = RK4(system, trajectory[i], dt)

    return trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]

# create time series for selected system
def get_time_series(system, initial_state, dt, steps, ts_transient):
    x_val, y_val, z_val = solve_system(system, initial_state, steps + 1, dt)
    data = np.vstack((x_val, y_val, z_val)).T
    return data[ts_transient:]


# ---------------------------------------------------------------------------------------------------
# Define Reservoir

# Parameters: nodes, probability of edge creation p = 0.01, seed = 42 and 15.000 reservoir iterations
n = 1000
p = 0.01
seed = 42

# generate reservoir network based on erdös-renyi network
network = nx.fast_gnp_random_graph(n=n, p=p, seed=seed)
network = nx.to_numpy_array(network)

# initialize weights of reservoir network
def initial_network():
    mask = network != 0  # Create a mask for non-zero entries
    network[mask] = np.random.uniform(-1, 1, size=np.count_nonzero(mask))
    return network

# rescale radius of eigenvalues
def scale_radius(rho=0.4, network=network):
    network = scipy.sparse.csr_matrix(network)
    eigenvalues = scipy.sparse.linalg.eigs(network, k=1, maxiter=int(1e3 * n))[0]

    maximum = np.absolute(eigenvalues).max()  # Maximum absolute value of eigenvalues
    network = rho / maximum * network  # Rescale network
    network = network.todense()  # Convert to dense matrix
    return network

# initialize weights of input matrix
def initial_w_in():
    w_in = np.random.default_rng(seed)
    w_in = w_in.uniform(low=-0.05, high=0.05, size=(n, 3))
    return w_in

# define regression to find output matrix
def ridge_reg(R,Y):
    regressor = Ridge(alpha=10**-8)
    regressor.fit(R,Y)
    W_out = regressor.coef_
    return W_out

# define reservoir state update function
def r_step(ri, A, w_in, xi):
    return np.tanh(A @ ri + w_in @ xi)

# define reservoir dynamics
def reservoir_dynamics(time_series, train_steps, r_transient):
    length = len(time_series)
    r = np.zeros(shape=(len(time_series), n))
    ts_pred = np.zeros(shape=(length - train_steps, 3))

    # training process
    for i in range(train_steps):
        r[i+1] = r_step(r[i], network, w_in, time_series[i+1])

    # regression to obtain output matrix using training process
    print(r)
    R = r[r_transient:train_steps+1]
    Y = time_series[r_transient+1:train_steps+2]
    w_out = ridge_reg(R, Y)

    # prediction process
    for enum, i in enumerate(np.arange(train_steps, length-1, 1)):
        ts_pred[enum] = w_out @ r[i]
        r[i+1] = r_step(r[i], network, w_in, ts_pred[enum])

    return ts_pred[:,0], ts_pred[:,1], ts_pred[:,2]


if __name__ == '__main__':
    # generate time series for Lorenz system
    ts = get_time_series(system=lorenz,
                                initial_state=np.array([1.0, 1.0, 1.0]),
                                dt=1e-2,
                                steps=100000,
                                ts_transient=5000)


    # generate time series for halverson system
    # ts = get_time_series(system=halverson,
    #                      initial_state=np.array([2.0, 1.0, 1.0]),
    #                      dt=1e-3,
    #                      steps=200000,
    #                      ts_transient=5000)

    # generate reservoir
    initial_network()
    network = scale_radius()
    w_in = initial_w_in()

    # define number of training steps to optimize output matrix
    train_steps = 50000

    # trigger reservoir dynamics and
    x_pred, y_pred, z_pred = reservoir_dynamics(time_series=ts,
                                                train_steps=train_steps,
                                                r_transient=5000)

    # Plot the trajectory
    ts_x, ts_y , ts_z = ts[train_steps:].T
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_pred[0], y_pred[0], z_pred[0], color='green', s=20, label='Starting Point of Prediction')
    ax.plot(x_pred, y_pred, z_pred, lw=2, color='blue', label='Reservoir Prediction')
    ax.plot(ts_x, ts_y , ts_z, color='red', linestyle='--', linewidth=0.5, label='Time Series of System')
    #ax.plot(-ts_x, -ts_y , -ts_z, color='green', linestyle='--', linewidth=0.5, label='Time Series of System')
    ax.set_title("Lorenz Attractor")
    ax.set_xlabel("X", fontsize=20)
    ax.set_ylabel("Y", fontsize=20)
    ax.set_zlabel("Z", fontsize=20)
    plt.legend()
    plt.show()

    # Plot the real and the predicted time series for the X,Y,Z coordinate
    xlim = 5000
    t_steps = np.arange(xlim)
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), constrained_layout=True)
    axs[0].plot(t_steps, x_pred[:xlim], label='[X]-Prediction')
    axs[0].plot(t_steps, ts_x[:xlim], color='red', label='[X]-Time-Series')
    axs[0].set_ylabel('X', fontsize=20)
    axs[1].plot(t_steps, y_pred[:xlim], label='[Y]-Prediction')
    axs[1].plot(t_steps, ts_y[:xlim], color='red', label='[Y]-Time-Series')
    axs[1].set_ylabel('Y', fontsize=20)
    axs[2].plot(t_steps, z_pred[:xlim], label='[Z]-Prediction')
    axs[2].plot(t_steps, ts_z[:xlim], color='red', label='[Z]-Time-Series')
    axs[2].set_ylabel('Z', fontsize=20)
    axs[0].legend(fontsize=15, loc='lower right')
    axs[1].legend(fontsize=15, loc='lower right')
    axs[2].legend(fontsize=15, loc='lower right')
    axs[2].set_xlabel(r'Time Step $[\Delta t]$', fontsize=20)
    plt.tight_layout()
    plt.show()

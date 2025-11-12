import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
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
# Define SINDY
# generate feature matrix of second order
def gen_feature_matrix(ts):
    if ts.ndim == 1:
        ts = [ts]

    feature_matrix = np.zeros(shape=(len(ts), 9))
    for enum, i in enumerate(ts):
        x,y,z = i[0], i[1], i[2]
        feature_matrix[enum] = np.array([x, y, z, x ** 2, y ** 2, z ** 2, x * y, x * z, y * z])
    return feature_matrix

# generate target data xi+1 - xi as approximation for ODE
def gen_target_data(ts):
    target_matrix = np.zeros(shape=(len(ts)-1, 3))
    for i in range(len(ts)-1):
        target_matrix[i] = ts[i+1] - ts[i]
    return target_matrix

# define regression to find matrix relation between input features and target data
def ridge_reg_and_pred(R,Y,x0,pred=None):
    regressor = Ridge(alpha=10**-8)
    regressor.fit(R,Y)
    W_out = regressor.coef_

    if pred:
        r = [R[-1]]
        x_pred = np.zeros(shape=(pred, 3))
        x_pred[0] = x0
        for i in range(pred - 1):
            x_pred[i+1] = regressor.predict(r) + x_pred[i]
            r = gen_feature_matrix(x_pred[i+1])
            regressor.predict([R[-1]])
        return W_out, x_pred

    else:
        return W_out, None

def SINDY(ts, pred, dt):
    if pred > len(ts):
        print(f'Length of time series is {len(ts)} but you chose {pred}, validation of prediction is '
              f'only possible within the length of the time series.')
    train = ts[:pred]
    val = ts[:pred]
    features = gen_feature_matrix(train)
    targets = gen_target_data(train)
    w_out, ts_pred = ridge_reg_and_pred(features[:-1], targets, ts[-1], pred=pred)
    w_out = (w_out / dt).round(decimals=1)
    return w_out, ts_pred, val


if __name__ == '__main__':
    dt_lorenz = 1e-3
    dt_halverson = 1e-3

    # generate time series for halverson system
    ts = get_time_series(system=halverson,
                         initial_state=np.array([2.0, 1.0, 1.0]),
                         dt=dt_halverson,
                         steps=50000,
                         ts_transient=5000)
    w_out, ts_pred, val = SINDY(ts, 51000, dt_halverson)


    # Plot the trajectory
    x_pred, y_pred, z_pred = ts_pred.T
    ts_x, ts_y , ts_z = val.T
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_pred[0], y_pred[0], z_pred[0], color='orange', s=30, label='Starting Point of Prediction')
    ax.plot(x_pred, y_pred, z_pred, lw=2, color='blue', label='Reservoir Prediction')
    ax.plot(ts_x, ts_y , ts_z, color='red', linestyle='--', linewidth=0.5, label='Time Series of System')
    ax.set_title("Lorenz Attractor")
    ax.set_xlabel("X", fontsize=20)
    ax.set_ylabel("Y", fontsize=20)
    ax.set_zlabel("Z", fontsize=20)
    plt.legend()
    plt.show()
    #
    # # Plot the real and the predicted time series for the X,Y,Z coordinate
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

import numpy as np
from tqdm import tqdm
from pathlib import Path
import os

# set parameters for lorenz system
a = 0.2
b = 0.2
c = 5
steps = 300
dt = 1e-1/3.0
noise_std = 0.03
initial_state = np.array([1.0,1.0,1.0])

def load_npz_folder(folder, ts):
    d = np.load(os.path.join(folder, ts))
    x = d["x"].astype(np.float32)
    y = d["y"].astype(np.float32)
    return x,y

def Rössler(x,y,z):
    dxdt = -y -z
    dydt = x + a*y
    dzdt = b + z*(x - c)
    return np.array([dxdt, dydt, dzdt])

def spiral(x, y, z):
    a_s = -0.1
    b_s = 1.0
    dxdt = a_s*x - b_s*y
    dydt = b_s*x + a_s*y
    dzdt = 0
    return np.array([dxdt, dydt, dzdt])

# define 4th order Runge-Kutta method
def RK4(system, state, dt):
    k1 = dt*system(*state)
    k2 = dt * system(*(state + 0.5 * k1))
    k3 = dt * system(*(state + 0.5 * k2))
    k4 = dt * system(*(state + k3))
    return state + (k1 + 2*k2 + 2*k3 + k4)/6

# Solve a given system with RK4
def solve_system(system, initial_state, steps, dt):
    trajectory = np.zeros((steps + 1, 3))
    trajectory[0] = initial_state

    for i in range(steps):
        trajectory[i+1] = RK4(system,trajectory[i],dt)

    if system.__name__ == "spiral":
        return np.array(trajectory[:, 0]), np.array(trajectory[:, 1]), np.array(np.zeros(len(trajectory[:, 2])))
    else:
        discard = 3*len(trajectory)//4
    return np.array(trajectory[discard:, 0]), np.array(trajectory[discard:, 1]), np.array(trajectory[discard:, 2])


def generate_time_series(system_name, system, initial_state, steps, std, dt, out_root=None):
    # Solve deterministic system
    x_val, y_val, z_val = solve_system(system, initial_state, steps, dt)
    # Output directory
    root = Path(__file__).parent if out_root is None else Path(out_root)
    out_dir = root / f"{system_name}_noise_std_{std}"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_path = out_dir / f"time_series.npz"
    np.savez_compressed(
        run_path,
        x=x_val,
        y=y_val,
        z=z_val,
        dt=np.array(dt),
        steps=np.array(steps),
        initial_state=np.array(initial_state),
        system_name=np.array(str(system_name)),
    )

    return run_path  # optional, handy for logging


def generate_measurements(system_name, std=0.1, num_measurements=10, out_root=None):
    # Solve deterministic system
    root = Path(__file__).parent if out_root is None else Path(out_root)
    out_dir = root / f"{system_name}_noise_std_{std}"
    out_dir.mkdir(parents=True, exist_ok=True)

    x_val, y_val = load_npz_folder(out_dir, f"time_series.npz")

    for i in range(num_measurements):
        arr = np.random.randint(1, steps, size=30)

        ts = np.sort(arr)

        x_m = x_val[ts] + np.random.normal(0, std, size=30)
        y_m = y_val[ts] + np.random.normal(0, std, size=30)

        # x_ts = x_ts[np.argsort(ts)]
        # y_ts = y_ts[np.argsort(ts)]
        run_path = out_dir / f"measurement_{i}_noise_std_{std}.npz"
        np.savez_compressed(
            run_path,
            x=np.array(x_m),
            y=np.array(y_m),
            t=ts*dt,
            system_name=np.array(str(system_name))
        )


generate_time_series("Spiral", spiral, [1.0, 0.0, 0.0], steps, noise_std, dt)
generate_measurements("Spiral", std=noise_std, num_measurements=10)
    
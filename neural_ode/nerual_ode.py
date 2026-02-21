import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import diffrax as dfx
import optax
import os 
import numpy as np
from tqdm import tqdm 


# ----------------------------
# 1) Neural vector field f_theta(x)
#    Must have signature (t, y, args) -> dy/dt for Diffrax
# ----------------------------
class VectorField(eqx.Module):
    net: eqx.nn.MLP

    def __init__(self, key, hidden=64):
        self.net = eqx.nn.MLP(
            in_size=2, out_size=2,
            width_size=hidden, depth=2,
            activation=jax.nn.tanh,
            key=key
        )

    def __call__(self, t, y, args):
        return self.net(y)   # autonomous: does not use t


# ----------------------------
# 2) ODE solve: integrate from y0 over times ts
# ----------------------------
def predict_trajectory(model, y0, ts, dt0=1e-2):
    term = dfx.ODETerm(model)
    sol = dfx.diffeqsolve(
        term,
        solver=dfx.Tsit5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=dt0,
        y0=y0,
        saveat=dfx.SaveAt(ts=ts),
        max_steps=10000,
        stepsize_controller=dfx.PIDController(
            rtol=1e-5,
            atol=1e-7,
        ),
        # You can set an adjoint explicitly if you like:
        # adjoint=dfx.RecursiveCheckpointAdjoint()
    )
    return sol.ys  # shape (len(ts), 2)

# ----------------------------
# 3) Loss: match the whole observed trajectory
# ----------------------------
def loss_fn(model, ts, ys, dt0):
    yhat = predict_trajectory(model, ys[0], ts, dt0=dt0)
    return jnp.mean(1/(5*1e-2)**2*(yhat - ys) ** 2)


# ----------------------------
# 4) One optimizer step (Adam)
# ----------------------------
@eqx.filter_jit
def train_step(model, opt_state, ts, ys, dt0, opt):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, ts, ys, dt0)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


# ----------------------------
# 5) Example training loop
#    You provide ys (T,2). If you don't have ts, build ts = dt * arange(T).
# ----------------------------
def train_neural_ode(ys, ts, dt=1e-2, hidden=32, lr=1e-3, steps=1000, seed=0):
    T = ys.shape[0]
    # ts = jnp.arange(T) * dt

    key = jr.PRNGKey(seed)
    model = VectorField(key, hidden=hidden)

    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    for k in tqdm(range(steps)):
        batch_size = T
        for batch in range(0, T, batch_size):
            batch_ys = ys[batch:batch+batch_size]
            batch_ts = ts[batch:batch+batch_size]
            model, opt_state, loss = train_step(model, opt_state, batch_ts, batch_ys, dt, opt)
        if k % 200 == 0:
            print(f"step {k:05d}  loss={float(loss):.6e}")

    return model


# ----------------------------
# 6) Predict X(t*) at an arbitrary time t*
# ----------------------------
def predict_at_time(model, y0, t_star, dt0=1e-2):
    ts = jnp.array([0.0, t_star])
    yhat = predict_trajectory(model, y0, ts, dt0=dt0)
    return yhat[-1]



def load_npz_folder(folder, ts=None):
    d = np.load(os.path.join(folder, "time_series.npz"))
    x0 = d["x"].astype(np.float32)[0]
    y0 = d["y"].astype(np.float32)[0]

    if ts is None:
        files = sorted(
            [
                f
                for f in os.listdir(folder)
                if f.startswith("measurement_") and f.endswith(".npz")
            ]
        )
        if not files:
            raise FileNotFoundError(f"No measurement_*.npz files found in {folder}")
    else:
        files = [ts]

    series = [jnp.array([[x0, y0]], dtype=jnp.float32)]  # shape (1,2)
    times = [jnp.array([0.0], dtype=jnp.float32)]
    for file in files:
        d = jnp.load(os.path.join(folder, file))
        x = d["x"].astype(jnp.float32)
        y = d["y"].astype(jnp.float32)
        ts = d["t"].astype(jnp.float32)
        series.append(jnp.stack([x, y], axis=-1))
        times.append(ts)

    series_all = jnp.concatenate(series, axis=0)   # (N, 2)
    times_all  = jnp.concatenate(times, axis=0)    # (N,)

    idx = jnp.argsort(times_all)                   # (N,)

    sorted_ts = times_all[idx]
    sorted_xy = series_all[idx]

    return sorted_xy, sorted_ts

def save_predictions(folder, ys, y_test, times):
    # Save test inputs, true outputs, and predictions to pred.npz
    pred_path = os.path.join(folder, "pred.npz")
    
    y_pred = []
    for time in times:
        y_pred.append(predict_at_time(model, ys[0], time))
    
    np.savez_compressed(
        pred_path,
        Y_true=np.asarray(ys),
        Y_pred=np.asarray(y_pred),
        Y_test=np.asarray(y_test),
    )
    print(f"Saved: {pred_path}")



# folder_rossler = "./FoAI_Interview/Rössler_data_noise_std_0.01/"
# data_rossler = "time_series_5f4ea5204f.npz"
# ts_rossler = load_npz_folder(folder_rossler, data_rossler)
# model = train_neural_ode(ts_rossler, steps=200, dt=1e-2)
# save_predictions(folder_rossler, ts_rossler)

folder_spiral = "./FoAI_Interview/Spiral_noise_std_0.03/"
# data_spiral = "measurement_0_noise_std_0.03.npz"
ys_spiral, t_spiral = load_npz_folder(folder_spiral)
test_split=int(0.3*len(t_spiral))
ys_train, t_train = ys_spiral[:-test_split], t_spiral[:-test_split]
y_test = ys_spiral[-test_split:]
model = train_neural_ode(ys_train, t_train, steps=2000, dt=1e-2)
save_predictions(folder_spiral, ys_spiral, y_test, t_spiral)

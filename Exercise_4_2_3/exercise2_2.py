import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as la
import scipy.sparse as sp
import time
import warnings
import generate

from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import Ridge

warnings.filterwarnings(action='ignore', category=RuntimeWarning)

delta_t = 10 ** -3


def least_squares(test, predictions):
    if len(test) != len(predictions):
        raise ValueError("Die Längen der beiden Arrays müssen gleich sein.")
    return np.sum((test - predictions) ** 2) / len(test)


def plot_trajectories(states, predictions, save, seed, name):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, projection='3d')  # 3D-Achsenobjekt erstellen

    # Originaldaten plotten
    ax.plot(states[:, 0], states[:, 1], states[:, 2], label="Original", color='red')
    # Vorhersagen plotten
    ax.plot(predictions[:, 0], predictions[:, 1], predictions[:, 2], label="Vorhersage", color='purple')

    # Achsenbeschriftungen
    ax.set_xlabel("x(t)")
    ax.set_ylabel("y(t)")
    ax.set_zlabel("z(t)")
    ax.set_title("Vergleich der Trajektorien")
    ax.legend()

    if save:  # Falls `save` angegeben ist, speichere die Grafik
        plt.savefig(
            f"/Users/julianblasek/master_local/complex_systems/4/plots/A2_RC_{name}_seed_{seed}_trajectories.pdf")
    plt.show()
    return -1


def plot_time_series(conv_states, predictions, save, seed, R, w_out, s, transient, n_train, name):
    t = np.arange(0, len(conv_states) * delta_t, delta_t)
    # Rekonstruiere die Zustände aus R
    reconstructed_train = R @ w_out.T  # Reservoir-Ausgabe auf den Zustandsraum zurückprojizieren

    plt.figure(figsize=(16, 8))

    # Subplot 1: Vergleich für x(t)
    plt.subplot(3, 1, 1)
    plt.plot(t, conv_states[:, 0], label="Lorenz-System", color='red')
    plt.plot(t[transient:n_train], reconstructed_train[:, 0], label="Reservoir (Training)", color='blue',
             linestyle='--')
    plt.plot(t[n_train:], predictions[:, 0], label="Reservoir (Vorhersage)", color='purple')
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.legend()
    plt.title("Vergleich für x(t)")

    # Subplot 2: Vergleich für y(t)
    plt.subplot(3, 1, 2)
    plt.plot(t, conv_states[:, 1], label="Lorenz-System", color='red')
    plt.plot(t[transient:n_train], reconstructed_train[:, 1], label="Reservoir (Training)", color='blue',
             linestyle='--')
    plt.plot(t[n_train:], predictions[:, 1], label="Reservoir (Vorhersage)", color='purple')
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    plt.title("Vergleich für y(t)")

    # Subplot 3: Vergleich für z(t)
    plt.subplot(3, 1, 3)
    plt.plot(t, conv_states[:, 2], label="Lorenz-System", color='red')
    plt.plot(t[transient:n_train], reconstructed_train[:, 2], label="Reservoir (Training)", color='blue',
             linestyle='--')
    plt.plot(t[n_train:], predictions[:, 2], label="Reservoir (Vorhersage)", color='purple')
    plt.xlabel("t")
    plt.ylabel("z(t)")
    plt.legend()
    plt.title("Vergleich für z(t)")

    # Anzeige der Plots
    plt.tight_layout()
    if save:
        plt.savefig(
            f"/Users/julianblasek/master_local/complex_systems/4/plots/A2_RC_{name}_seed_{seed}_time_series.pdf")
    plt.show()

    return -1


def generate_future_states(n, predictions, R_next, w_out, w_res, w_in, n_gen):
    start = time.time()
    R_next_add = np.zeros((n_gen, n))
    predictions_add = np.zeros((n_gen, 3))
    R_next_add[0] = R_next[-1]

    for i in range(n_gen - 1):
        predictions_add[i] = w_out @ R_next_add[i]
        R_next_add[i + 1] = next_state(w_res, R_next_add[i], w_in, predictions_add[i])

    print("-Future States Trained: Time: ", np.round(time.time() - start, 3), "s")
    return np.concatenate((predictions, predictions_add))


def vergleich(test, predictions, save, seed, name):
    t = np.arange(0, len(test) * delta_t, delta_t)
    plt.figure(figsize=(16, 8))

    # Subplot 1: Vergleich für x(t)
    plt.subplot(3, 1, 1)  # 3 Zeilen, 1 Spalte, erster Subplot
    plt.plot(t, test[:, 0], label="Original", color='red')
    plt.plot(t, predictions[:, 0], label="Vorhersage", color='purple')
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.legend()
    plt.title("Vergleich für x(t)")

    # Subplot 2: Vergleich für y(t)
    plt.subplot(3, 1, 2)  # 3 Zeilen, 1 Spalte, zweiter Subplot
    plt.plot(t, test[:, 1], label="Original", color='red')
    plt.plot(t, predictions[:, 1], label="Vorhersage", color='purple')
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    plt.title("Vergleich für y(t)")

    # Subplot 3: Vergleich für z(t)
    plt.subplot(3, 1, 3)  # 3 Zeilen, 1 Spalte, dritter Subplot
    plt.plot(t, test[:, 2], label="Original", color='red')
    plt.plot(t, predictions[:, 2], label="Vorhersage", color='purple')
    plt.xlabel("t")
    plt.ylabel("z(t)")
    plt.legend()
    plt.title("Vergleich für z(t)")

    # Anzeige der Plots
    plt.tight_layout()  # Passt die Abstände zwischen den Subplots an
    if save:
        plt.savefig(f"/Users/julianblasek/master_local/complex_systems/4/plots/A2_RC_{name}_seed_{seed}_comparison.pdf")
    plt.show()


def plot_network(network):
    plt.figure(figsize=(12, 8))
    nx.draw(network, with_labels=True, node_size=1)
    plt.show()


#   W_reservoir
def init_res(n, p, seed, s_rad):
    network = nx.to_numpy_array(nx.fast_gnp_random_graph(n, p, seed=seed))
    for row_idx, row in enumerate(network):
        for col_idx, value in enumerate(row):
            if value != 0:
                network[row_idx, col_idx] = np.random.uniform(-1, 1)

    # Spectral Radius
    network = sp.csr_matrix(network)
    eigenvalues = la.eigs(network, k=1, maxiter=int(1e3 * n))[0]

    maximum = np.absolute(eigenvalues).max()  # Maximum absolute value of eigenvalues
    network = s_rad / maximum * network  # Rescale network
    network = network.todense()  # Convert to dense matrix
    return network


def next_state(w_res, r, w_in, x):
    return np.tanh(w_res @ r + w_in @ x)


def iterate_reservoir(w_res, w_in, train):
    n = len(w_res)
    r_0 = np.zeros(n)  # Initial state of the reservoir (N,)
    R = np.zeros((len(train), n))  # Reservoir states (15000,N)
    R[0] = r_0

    for i in range(1, len(train)):
        R[i] = next_state(w_res, R[i - 1], w_in, train[i])
    return R


def main(alpha, save, n, p, s_rad, seed, n_train, n_test, n_future, transient, f, name):
    start = time.time()

    # Initialisierung der Reservoirs
    w_res_list = []
    for s in s_rad:
        w_res_list.append(init_res(n, p, seed, s))
    w_in = np.random.default_rng(seed).uniform(-0.05, 0.05, (n, 3))  # n: Zeilen, 3 Spalten (x,y,z)-> Nx3

    # Generierung der Trainings- und Testdaten
    states, conv_states = f(n_train + n_test)
    train = conv_states[:n_train]
    test = conv_states[n_train:]

    # Iterate Reservoir for diff. spectral radii
    R_list = []
    for w_res in w_res_list:
        R_list.append(iterate_reservoir(w_res, w_in, train))

    regressor_list = []
    for i in alpha:
        regressor = Ridge(alpha=i)
        regressor_list.append(regressor)

    y_train = train[transient + 1:]  # x_i+1 (Target)

    least_squares_list = []
    s_c = 0
    for R, w_res in zip(R_list, w_res_list):
        print("-Train Regressor for spectral radius: ", s_rad[s_c], " -")
        s_c += 1
        R = R[transient:]  # r_i (Input)
        w_out_list = []
        for regressor in regressor_list:
            regressor.fit(R[:-1], y_train)  # r_i (Input) und x_i+1 (Target)
            w_out_list.append(regressor.coef_)

        print("-Regressors Trained : Time: ", np.round(time.time() - start, 3), "s")

        ij = 0
        for w_out in w_out_list:
            print("-Predictions for Alpha: ", alpha[ij], "-")
            ij = ij + 1
            start = time.time()
            predictions = np.zeros((len(test), 3))
            R_next = np.zeros((len(test), n))
            R_next[0] = R[-1]
            for i in range(len(test) - 1):
                predictions[i] = w_out @ R_next[i]
                R_next[i + 1] = next_state(w_res, R_next[i], w_in, predictions[i])

            least_squares_list.append(least_squares(test, predictions))
            predictions_2 = generate_future_states(n, predictions, R_next, w_out, w_res, w_in, n_future)
            plot_trajectories(states, predictions_2, save=save, seed=seed, name=name)

            # vergleich(test, predictions,save=save,seed=seed,name=name)
            # plot_time_series(conv_states,predictions, save=save, seed=seed, R=R, w_out=w_out,s=s,transient=transient,n_train=n_train,name=name)
            # plot_trajectories(conv_states, predictions, save=save, seed=seed,name=name)

    s_c = 0  # Counter für den spektralen Radius
    for regressor_index, regressor in enumerate(regressor_list):  # Iteriere über alle Alpha-Werte
        for s_rad_index, s_rad_value in enumerate(s_rad):  # Iteriere über alle spektralen Radien
            print(
                f"Least Squares: {least_squares_list[s_c]} for spec. Radius: {s_rad_value}, Alpha: {alpha[regressor_index]}")
            s_c += 1

    return -1


plt.ion()
alpha = [10 ** -8]
s_rad_l = [0.4]
s_rad_h = [0.57]
for i in [42]:
    main(alpha, save=False, n=1000, p=0.01, s_rad=s_rad_l, seed=i, n_train=35000, n_test=10000, n_future=50000,
         transient=2000, f=generate.generate_lorenz, name="Lorenz")
    # main(alpha,save=False,n=1000,p=0.01,s_rad=s_rad_h,seed=i,n_train=35000,n_test=10000,n_future=50000,transient=2000,f=generate.generate_halvorsen,name="Halvorsen")
plt.ioff()
plt.show()




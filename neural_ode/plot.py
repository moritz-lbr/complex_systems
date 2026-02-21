import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def load_test_predictions(folder, filename="pred.npz"):
    """
    Loads pred.npz from `folder` and returns:
      x_pred, y_pred, z_pred, x_true, y_true, z_true

    Expects pred.npz to contain:
      - Y_test: (N, 3)
      - Y_pred: (N, 3)
    """
    path = os.path.join(folder, filename)
    data = np.load(path)


    Y_true = data["Y_true"]  # shape (N, 3)
    Y_pred = data["Y_pred"]  # shape (N, 3)
    Y_test = data["Y_test"]  # shape (N, 3)

    x_pred = Y_pred[:, 0]
    y_pred = Y_pred[:, 1]
    # z_pred = Y_pred[:, 2]

    x_true = Y_true[:, 0]
    y_true = Y_true[:, 1]
    # z_true = Y_true[:, 2]

    x_test = Y_test[:, 0]
    y_test = Y_test[:, 1]

    # return x_pred, y_pred, z_pred, x_true, y_true, z_true
    return x_pred, y_pred, x_true, y_true, x_test, y_test


# x_pred, y_pred, z_pred, x_true, y_true, z_true = load_test_predictions(
#     "./FoAI_Interview/Rössler_data_noise_std_0.01"
# )

def load_npz_folder(folder, ts):
    d = np.load(os.path.join(folder, ts))
    x = d["x"].astype(np.float32)
    y = d["y"].astype(np.float32)
    return x,y

folder = "./FoAI_Interview/Spiral_noise_std_0.03"
x_ts, y_ts = load_npz_folder(folder, "time_series.npz")
x_pred, y_pred, x_true, y_true, x_test, y_test = load_test_predictions(folder)


fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(x_ts, y_ts, label="Particle Trajectory", color='black', lw=1.0)
ax.plot(x_pred, y_pred, marker='D', markersize=5, color='green', lw=1.0)
ax.scatter(x_true, y_true, s=20,  color="gray")
ax.scatter(x_test, y_test, s=20,  color="purple")
# ax.scatter(x_true[0], y_true[0], s=100, color='red',
#            label=r"Initial Position $X(0) = 0$")

legend_handles = []
legend_handles.append(Line2D([0], [0], marker='o', markersize=10, linestyle='None', color="gray", label=f"Combined Measurements" +  r" $\{(X_i, Y_i)\}_{i=1}^{T}$"))
# legend_handles.append(Line2D([0], [0], marker='o', linestyle='None', color='red',markersize=10, label=r"Initial Position $X(0)$"))
legend_handles.append(Line2D([0], [0], marker=None, linestyle='-', linewidth=5, color='black', markersize=10, label=r"Particle Trajectory"))
legend_handles.append(Line2D([0], [0], marker=None, linestyle='-', linewidth=5, color='purple', markersize=10, label=r"Test Measurements"))
legend_handles.append(Line2D([0], [0], marker="D", linestyle='-', linewidth=5, color='green', markersize=10, label=r"Neural ODE Trajectory"))


# Add a "text-box legend" (no frame around the axes; the box is the legend itself)
leg = plt.gca().legend(
    handles=legend_handles,
    loc="lower right",          # change as needed
    frameon=True,               # makes it a boxed text area
    fancybox=True,
    framealpha=0.9,
    borderpad=0.8,
    handletextpad=0.6,
    fontsize=20
)


# Optional: tweak box edge/background
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(1.0)

ax.set_title(r"True Particle Trajectory and Noisy Measurements", fontsize=30)
ax.set_xlabel("X", fontsize=30)
ax.set_ylabel("Y", fontsize=30)
ax.tick_params(axis='both', labelsize=15)
plt.grid()
plt.tight_layout()
plt.show()








# Plot the trajectory
# slice = -1
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
# # ax.plot(x_pred[:slice], y_pred[:slice], z_pred[:slice], lw=0.5, color='blue', label="Predicted")
# # ax.plot(x_true[:slice], y_true[:slice], z_true[:slice], lw=0.5, color='black', label="True")
# # ax.scatter(x_pred[0], y_pred[0], z_pred[0], color='red', s=10)
# ax.plot(x_pred[:slice], y_pred[:slice], lw=0.5, color='blue', label="Predicted")
# ax.scatter(x_true[:slice], y_true[:slice], lw=0.5, color='black', label="True")
# ax.scatter(x_pred[0], y_pred[0], color='red', s=10)
# ax.set_title("Spiral")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# plt.show()
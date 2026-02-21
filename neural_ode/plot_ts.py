import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def load_npz_folder(folder, ts):
    d = np.load(os.path.join(folder, ts))
    x = d["x"].astype(np.float32)
    y = d["y"].astype(np.float32)
    return x,y

x_true, y_true = load_npz_folder("./FoAI_Interview/Spiral_noise_std_0.03", "time_series.npz")

num_measurements = 1

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(x_true, y_true, label="Particle Trajectory", color='black', lw=1.0)

cmap = plt.get_cmap("tab20")          # good for distinct lines; try "tab20" for more
colors = cmap(np.linspace(0, 1, num_measurements)) 

legend_handles = []
for i in range(num_measurements):
    x_noise, y_noise = load_npz_folder("./FoAI_Interview/Spiral_noise_std_0.03", f"measurement_{i}_noise_std_0.03.npz")
    ax.scatter(x_noise, y_noise, s=40,  color=colors[i])
    legend_handles.append(Line2D([0], [0], marker='o', markersize=10, linestyle='None', color=colors[i], label=f"Measurement {i+1}" +  r" $\{(X_i, Y_i)\}_{i=1}^{T}$"))
    
ax.scatter(x_true[0], y_true[0], s=100, color='red',
           label=r"Initial Position $X(0) = 0$")

legend_handles.append(Line2D([0], [0], marker='o', linestyle='None', color='red',
                    markersize=10, label=r"Initial Position $X(0)$"))
legend_handles.append(Line2D([0], [0], marker=None, linestyle='-', color='black',
                    markersize=10, label=r"Particle Trajectory"))

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
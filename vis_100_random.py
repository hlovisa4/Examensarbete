import os
import random
import laspy
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("Agg")  # headless

input_folder = "C:/Users/digit/Downloads/Examensarbete/ff3d_tls_als_trees"
output_folder = "C:/Users/digit/Downloads/Examensarbete/Results/TLS_ALS_v_ff3d_visualization_100_random"
os.makedirs(output_folder, exist_ok=True)

def las_to_np(file_path):
    las = laspy.read(file_path)
    return np.vstack((las.x, las.y, las.z)).T

n_samples = 100
las_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".las")]
random.seed(42) 
for filename in random.sample(las_files, min(n_samples, len(las_files))):
    pts = las_to_np(os.path.join(input_folder, filename))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # XZ projection (X vs Z, colored by Y)
    ax1.scatter(pts[:, 0], pts[:, 2], s=0.2, c=pts[:, 1], cmap="viridis", marker=".")
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title("XZ Projection")

    # YZ projection (Y vs Z, colored by X)
    ax2.scatter(pts[:, 1], pts[:, 2], s=0.2, c=pts[:, 0], cmap="viridis", marker=".")
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title("YZ Projection")

    png_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
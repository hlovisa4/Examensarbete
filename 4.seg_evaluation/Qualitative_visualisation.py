import os
import random
import re
import laspy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend("Agg")  # headless

input_folder = "C:/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/ALS_TLS_extracted_trees_ff3d"
output_folder = "C:/Users/digit/Downloads/Examensarbete/Results/Qual_eval/TLS_ALS_FF3D/"
os.makedirs(output_folder, exist_ok=True)

ref = pd.read_csv("C:/Users/digit/Downloads/Examensarbete/Results/Qual_eval/random_100_all_matched.csv")
ids = set(ref["ff3d_tile_id"].astype(str)) 
ids = {str(int(float(x))) for x in ids}
print(ids)
all_files = os.listdir(input_folder)

file_list = []

pattern = re.compile(r"_i_(\d+)_")

for f in all_files:
    match = pattern.search(f)
    if match:
        file_id = match.group(1)
        if file_id in ids:
            file_list.append(f)

print(len(file_list))

def las_to_np(file_path):
    las = laspy.read(file_path)
    return np.vstack((las.x, las.y, las.z)).T


for filename in file_list:
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
    print(f"Saved visualization for {filename} to {png_path}")
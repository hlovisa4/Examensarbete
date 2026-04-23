import os
import random
import laspy
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

plt.switch_backend("Agg")  # headless

input_folder = "C:/Users/digit/Downloads/Examensarbete/ff3d_tls_als_trees"
output_folder = "C:/Users/digit/Downloads/Examensarbete/Results/TLS_ALSff3d_visualization_100_random"
os.makedirs(output_folder, exist_ok=True)
gdf = gpd.read_file("C:/Users/digit/Downloads/Examensarbete/Results/matched_trees.gpkg")
filtered_gdf = gdf[gdf['ff3d_id'].notna() & gdf['chm_id'].notna()]
random_sample = filtered_gdf.sample(n=100, random_state=42)
sample_ids = set(random_sample['ff3d_id'].astype(str))

def las_to_np(file_path):
    las = laspy.read(file_path)
    return np.vstack((las.x, las.y, las.z)).T

las_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".las")]
matched_files = [
    f for f in las_files 
    if any(f"i_{id_val}" in f for id_val in sample_ids)
]

print(f"Found {len(matched_files)} matching .las files.")


random.seed(42) 
for filename in matched_files:
    pts = las_to_np(os.path.join(input_folder, filename))
    print(pts[2])
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
import os
import random
import laspy
import numpy as np
import open3d as o3d

# --- CONFIG ---
input_folder = "path/to/las_files"
output_folder = "path/to/output_pngs"
os.makedirs(output_folder, exist_ok=True)

n_samples = 100  # number of LAS files to process

# --- FUNCTION TO CONVERT LAS TO OPEN3D POINT CLOUD ---
def las_to_o3d(file_path):
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

# --- GET RANDOM SAMPLE OF LAS FILES ---
las_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".las")]
selected_files = random.sample(las_files, min(n_samples, len(las_files)))

print(f"Processing {len(selected_files)} randomly selected LAS files...")

# --- LOOP THROUGH SELECTED FILES ---
for filename in selected_files:
    las_path = os.path.join(input_folder, filename)
    print(f"Processing {filename} ...")

    pcd = las_to_o3d(las_path)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    png_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")
    vis.capture_screen_image(png_path)
    vis.destroy_window()

    print(f"Saved screenshot to {png_path}")

print("Done.")
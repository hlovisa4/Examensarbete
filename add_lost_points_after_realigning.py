import os
import laspy
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from tqdm import tqdm

INPUT_FOLDER = "C:/Users/digit/Downloads/Examensarbete/Data/las_polygon/240829_ALS_Matrice300_Svb_clipped.las"
OUTPUT_FILE = "C:/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/remerged_ff3d_segmented_cloud.las"

# Use a small tolerance to match identical points
MATCH_TOL = 1e-3  

# -----------------------------
# LOAD ORIGINAL POINTS
# -----------------------------
print("Loading ORIGINAL points...")
orig_points = []


las = laspy.read(INPUT_FOLDER)
pts = np.vstack((las.x, las.y, las.z)).T
orig_points.append(pts)

orig_points = np.vstack(orig_points)

# -----------------------------
# LOAD OUTPUT (KEPT) POINTS
# -----------------------------
print("Loading OUTPUT points...")
out_points = []

las = laspy.read(OUTPUT_FILE)
pts = np.vstack((las.x, las.y, las.z)).T
out_points.append(pts)

out_points = np.vstack(out_points)

print(f"Original points: {len(orig_points)}")
print(f"Output points:   {len(out_points)}")

# -----------------------------
# BUILD KD-TREE ON OUTPUT
# -----------------------------
print("Building KD-tree...")
tree = cKDTree(out_points)

# For each original point, check if it exists in output
distances, _ = tree.query(orig_points, k=1)

# Points NOT in output (i.e., missing)
missing_mask = distances > MATCH_TOL
missing_points = orig_points[missing_mask]

print(f"Missing points: {len(missing_points)}")


header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
    
# Preserve scale and offset (important!)
header.scales = las.header.scales
header.offsets = las.header.offsets

# Create LAS dataset
las_out = laspy.LasData(header)

las_out.x = missing_points[:, 0]
las_out.y = missing_points[:, 1]
las_out.z = missing_points[:, 2]
for dim in las.point_format.dimension_names:
   print(dim)
   if dim not in ["X", "Y", "Z"]:
         las_out[dim] = np.zeros(len(missing_points), dtype=las[dim].dtype)

las_out.write("C:/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/missing_points.las")
import laspy
import numpy as np
from laspy import ExtraBytesParams

# -----------------------------
# INPUTS
# -----------------------------
TLS_FILE = "C:/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/TLS_labeled_from_ALS_ff3d.las"
ALS_FILE = "C:/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/remerged_ff3d_segmented_cloud_plus_missing_points.las"
OUTPUT_FILE = "C:/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/merged_tls_als.las"

# -----------------------------
# LOAD FILES
# -----------------------------
print("Loading LAS files...")
tls = laspy.read(TLS_FILE)
als = laspy.read(ALS_FILE)

# -----------------------------
# CHECK POINT FORMAT COMPATIBILITY
# -----------------------------
tls_dims = set(tls.point_format.dimension_names)
als_dims = set(als.point_format.dimension_names)

common_dims = list(tls_dims.intersection(als_dims))

if tls_dims != als_dims:
    print("Warning: dimension mismatch detected")
    print(f"TLS only: {tls_dims - als_dims}")
    print(f"ALS only: {als_dims - tls_dims}")
    print("-> Keeping only common dimensions")

# -----------------------------
# CREATE NEW HEADER
# -----------------------------
print("Creating new header...")
header = tls.header.copy()

# Add TLS flag dimension
header.add_extra_dim(ExtraBytesParams(name="tls", type=np.uint8))

# -----------------------------
# CREATE OUTPUT LAS
# -----------------------------
merged = laspy.LasData(header)

# -----------------------------
# MERGE DIMENSIONS
# -----------------------------
print("Merging point data...")

for dim in common_dims:
    merged[dim] = np.concatenate([tls[dim], als[dim]])

# -----------------------------
# ADD TLS FLAG
# -----------------------------
print("Adding TLS flag...")

tls_flag_tls = np.ones(len(tls.points), dtype=np.uint8)
tls_flag_als = np.zeros(len(als.points), dtype=np.uint8)

merged["tls"] = np.concatenate([tls_flag_tls, tls_flag_als])

# -----------------------------
# HANDLE SCALES & OFFSETS
# -----------------------------
print("Aligning scales and offsets...")

merged.header.scales = tls.header.scales
merged.header.offsets = tls.header.offsets

# -----------------------------
# SAVE OUTPUT
# -----------------------------
print("Writing output LAS...")
merged.write(OUTPUT_FILE)

print(f"Done! Saved to: {OUTPUT_FILE}")
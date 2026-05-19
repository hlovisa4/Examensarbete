import os
import laspy
import numpy as np
from shapely.geometry import Point
from shapely import contains_xy
from tqdm import tqdm

# -----------------------------
# INPUT LAS FILE
# -----------------------------
tls_path = r"/mnt/c/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/tls_als_ff3d_final.las"

# -----------------------------
# OUTPUT DIRECTORY
# -----------------------------
out_dir = r"/mnt/c/Users/digit/Downloads/Examensarbete/Results/saptrees/"
os.makedirs(out_dir, exist_ok=True)

# -----------------------------
# CIRCLE CENTERS (SWEREF99 TM)
# -----------------------------
centers = [
    ("tree408", 731337.678, 7134016.31),
    ("tree403", 731341.472, 7134019.67),
]

radius = 2.5  # meters
chunk_size = 1_000_000

# -----------------------------
# CREATE CIRCULAR POLYGONS
# -----------------------------
circles = {
    name: Point(x, y).buffer(radius)
    for name, x, y in centers
}

# -----------------------------
# OPEN INPUT LAS
# -----------------------------
with laspy.open(tls_path) as reader:

    header = reader.header

    # Open one writer for each output file
    writers = {}
    for name in circles:
        out_path = os.path.join(out_dir, f"{name}_5m.las")
        writers[name] = laspy.open(out_path, mode="w", header=header)

    # Use context managers
    with writers["tree408"] as writer408, writers["tree403"] as writer403:

        writer_map = {
            "tree408": writer408,
            "tree403": writer403,
        }

        # Iterate through LAS in chunks
        for pts in tqdm(reader.chunk_iterator(chunk_size)):

            x = np.asarray(pts.x)
            y = np.asarray(pts.y)

            # Clip points for each circle
            for name, polygon in circles.items():

                mask = contains_xy(polygon, x, y)

                if np.any(mask):
                    writer_map[name].write_points(pts[mask])

print("Done: Created two clipped LAS files with 5 m radius.")
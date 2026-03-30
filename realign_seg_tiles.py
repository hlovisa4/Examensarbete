import os
import re
import laspy
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import networkx as nx
from tqdm import tqdm

# -----------------------------
# PARAMETERS
# -----------------------------
TILE_SIZE = 20
BUFFER = 7
DIST_THRESHOLD = 2.0  # meters (matching threshold)

INPUT_FOLDER = "C:/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/output"
OUTPUT_FILE = "C:/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation"

# -----------------------------
# HELPERS
# -----------------------------
def parse_tile_origin(filename):
    match = re.match(r"(\d+)_(\d+)", filename)
    if match:
        return float(match.group(1)), float(match.group(2))
    else:
        raise ValueError(f"Cannot parse coordinates from {filename}")
    

def compute_centroid(x, y):
    return np.mean(x), np.mean(y)

def is_core(cx, cy, xmin, ymin):
    return (
        xmin + BUFFER <= cx <= xmin + TILE_SIZE - BUFFER and
        ymin + BUFFER <= cy <= ymin + TILE_SIZE - BUFFER
    )

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# -----------------------------
# STEP 1 — LOAD ALL INSTANCES
# -----------------------------
instances = []
instance_id_counter = 0

print("Loading LAS tiles...")

for file in (os.listdir(INPUT_FOLDER)): #tqdm
    if not file.endswith(".las"):
        continue

    filepath = os.path.join(INPUT_FOLDER, file)
    xmin, ymin = parse_tile_origin(file)

    las = laspy.read(filepath)
    tree_ids = np.unique(las.instance_pred)

    for tid in tree_ids:
        if tid == 0 or tid == -1:
            continue  # skip background if exists

        mask = las.instance_pred == tid
        x = las.x[mask]
        y = las.y[mask]

        if len(x) < 30:
            continue  # filter tiny noise

        cx, cy = compute_centroid(x, y)

        inst = {
            "global_id": instance_id_counter,
            "tile": file,
            "treeID": tid,
            "centroid": (cx, cy),
            "point_count": len(x),
            "is_core": is_core(cx, cy, xmin, ymin),
            "xmin": xmin,
            "ymin": ymin,
        }

        instances.append(inst)
        instance_id_counter += 1

print(f"Total instances: {len(instances)}")

# -----------------------------
# STEP 2 — BUILD KD-TREE
# -----------------------------
centroids = np.array([inst["centroid"] for inst in instances])
tree = cKDTree(centroids)

# -----------------------------
# STEP 3 — FIND MATCHES
# -----------------------------
print("Matching overlapping instances...")

edges = []

for i, inst in enumerate(tqdm(instances)):
    neighbors = tree.query_ball_point(inst["centroid"], DIST_THRESHOLD)

    for j in neighbors:
        if i >= j:
            continue

        inst2 = instances[j]

        # Only match across different tiles
        if inst["tile"] == inst2["tile"]:
            continue

        d = distance(inst["centroid"], inst2["centroid"])

        if d < DIST_THRESHOLD:
            edges.append((i, j))

print(f"Total matches: {len(edges)}")

# -----------------------------
# STEP 4 — BUILD GRAPH
# -----------------------------
G = nx.Graph()
G.add_nodes_from(range(len(instances)))
G.add_edges_from(edges)

components = list(nx.connected_components(G))

print(f"Found {len(components)} tree clusters")

# -----------------------------
# STEP 5 — SELECT BEST INSTANCE
# -----------------------------
#Här skulle man kunna utveckla selektionsalgoritmen, vill jag tex spara bara de instanser med högst point count
def score(inst):
    return (
        inst["is_core"] * 1000 +
        inst["point_count"]
    )

selected_instances = []

for comp in components:
    comp_list = [instances[i] for i in comp]
    best = max(comp_list, key=score)
    selected_instances.append(best)

print(f"Selected {len(selected_instances)} final trees")

# -----------------------------
# STEP 6 — ASSIGN GLOBAL IDs
# -----------------------------
#for new_id, inst in enumerate(selected_instances, start=1):
#    inst["treeID"] = new_id

# Create lookup
selected_lookup = {
    (inst["tile"], inst["treeID"]): new_id
    for new_id, inst in enumerate(selected_instances, start=1)
}

# -----------------------------
# STEP 7 — MERGE POINT CLOUDS
# -----------------------------
print("Merging LAS files...")

all_points = []
total_points = 0
kept_points = 0
discarded_points = 0

for file in tqdm(os.listdir(INPUT_FOLDER)):

    if not file.endswith(".las"):
        continue

    filepath = os.path.join(INPUT_FOLDER, file)
    las = laspy.read(filepath)

    total_points += len(las.x)

    mask = np.zeros(len(las.x), dtype=bool)
    new_ids = np.zeros(len(las.x), dtype=int)

    for tid in np.unique(las.instance_pred):
        key = (file, tid)

        if key not in selected_lookup:
            discarded_points += np.sum(las.instance_pred == tid)
            continue

        m = las.instance_pred == tid
        mask |= m
        new_ids[m] = selected_lookup[key]
        kept_points += np.sum(m)

    if np.sum(mask) == 0:
        continue

    pts = las.points[mask].copy()
    pts["instance_pred"] = new_ids[mask]
    out_path =  f"{OUTPUT_FILE}/{file[:-4]}_saved_insts.las"
    with laspy.open(out_path, mode="w", header=las.header.copy()) as writer:
        writer.write_points(pts)
    
    all_points.append(pts)

print(f"\nPoint statistics:")
print(f"  Total points: {total_points}")
print(f"  Kept points: {kept_points}")
print(f"  Discarded points: {discarded_points}")
print(f"  Retention rate: {100*kept_points/total_points:.1f}%")
# -----------------------------
# STEP 8 — SAVE OUTPUT
# -----------------------------
print("Saving merged LAS...")
las_orig = laspy.read("C:/Users/digit/Downloads/Examensarbete/Data/las_polygon/240829_ALS_Matrice300_Svb_clipped.las")
header = las_orig.header.copy() 
header.add_extra_dim(laspy.ExtraBytesParams(name="semantic_pred", type=np.int32))
header.add_extra_dim(laspy.ExtraBytesParams(name="score", type=np.int32))
header.add_extra_dim(laspy.ExtraBytesParams(name="semantic_gt", type=np.int32))
header.add_extra_dim(laspy.ExtraBytesParams(name="instance_gt", type=np.int32))



#with laspy.open(OUTPUT_FILE, mode="w", header=header) as writer:
#    for points in all_points:
        #new_points = laspy.ScaleAwarePointRecord.zeros(len(points), header=header)

        #for dim in points.point_format.dimension_names:
        #            # avoid copying dims that are extras only in output
        #            if dim in new_points.point_format.dimension_names:
        #                new_points[dim] = points[dim]
#        writer.write_points(points)


print(f"Done! Saved to {OUTPUT_FILE}")
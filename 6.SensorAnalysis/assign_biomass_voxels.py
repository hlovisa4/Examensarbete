from ast import If
from collections import defaultdict

import matplotlib
import numpy as np
import pandas as pd
import open3d as o3d
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from calculate_biomass import MarklundBiomass
from plyfile import PlyData

#Requires biomass_venv!!

VOXEL_SIZES = [0.1, 0.2, 0.4]
COVERAGE_THRESHOLD = 0.6

HEIGHT_BINS = [
    (0.0, 1.3),
    (1.3, 8.0),
    (8.0, 14.0),
    (14.0, np.inf)
]

def load_xyz(ply_path1):
    pcd1 = o3d.io.read_point_cloud(str(ply_path1))
    return np.asarray(pcd1.points)

def voxelize(points, voxel_size):
    if len(points) == 0:
        return {}

    coords = np.floor(points / voxel_size).astype(int)

    unique, counts = np.unique(coords, axis=0, return_counts=True)
    voxel_counts = {tuple(u): c for u, c in zip(unique, counts)}

    return voxel_counts

def distribute_biomass(voxel_counts, total_biomass):
    if not voxel_counts:
        return {}

    counts = np.array(list(voxel_counts.values()))
    total_points = counts.sum()

    biomass_per_voxel = {}
    for voxel, count in voxel_counts.items():
        biomass_per_voxel[voxel] = (total_biomass / len(voxel_counts)) 
    return biomass_per_voxel


def aggregate_height_bins(biomass_voxels, voxel_size):
    bins = np.zeros(len(HEIGHT_BINS))
    layers = defaultdict(list)
    for (ix, iy, iz), biomass in biomass_voxels.items():
        # use voxel center
        z = (iz + 0.5) * voxel_size
        layers[z].append(((ix + 0.5) * voxel_size, (iy + 0.5) * voxel_size))
        for i, (zmin, zmax) in enumerate(HEIGHT_BINS):
            if zmin <= z < zmax:
                bins[i] += biomass
                break

    return layers, bins

def voxels_to_dataframe(voxel_dict, voxel_size, tree_id, component, stem_centers):
    rows = []

    for (ix, iy, iz), biomass in voxel_dict.items():
        direction = None

        if component == "foliage" and stem_centers is not None:
            #print(f"Assigning direction for voxel ({ix}, {iy}, {iz}) with biomass {biomass:.2f}")
            direction = assign_direction(
                ix, iy, iz,
                stem_centers,
                voxel_size
            )
        rows.append({
            "tree_id": tree_id,
            "voxel_size": voxel_size,
            "x": (ix + 0.5) * voxel_size,
            "y": (iy + 0.5) * voxel_size,
            "z": (iz + 0.5) * voxel_size,
            "ix": ix,
            "iy": iy,
            "iz": iz,
            "biomass": biomass,
            "component": component,
            "direction": direction
        })

    return pd.DataFrame(rows)

def compute_stem_centers(stem_voxels, voxel_size):
    """
    Computes stem center (x,y) for each z-slice (iz).
    Uses voxel centers.
    """

    stem_slices = defaultdict(list)

    for (ix, iy, iz) in stem_voxels.keys():
        x = (ix + 0.5) * voxel_size
        y = (iy + 0.5) * voxel_size

        stem_slices[iz].append((x, y))

    stem_centers = {}

    for iz, pts in stem_slices.items():
        pts = np.array(pts)

        stem_centers[iz] = {
            "x": pts[:, 0].mean(),
            "y": pts[:, 1].mean()
        }

    return stem_centers

def assign_direction(ix, iy, iz, stem_centers, voxel_size):
    """
    Assign N/S direction based on foliage position
    relative to stem center at same height.

    If no stem exists at iz:
    searches downward until one is found.
    """

    foliage_y = (iy + 0.5) * voxel_size

    search_iz = iz

    while search_iz >= 0:

        if search_iz in stem_centers:

            stem_y = stem_centers[search_iz]["y"]

            if foliage_y >= stem_y:
                return "N"
            else:
                return "S"

        search_iz -= 1

    return None

# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------
def process_tree(stem_path, canopy_path, csv_path, id, voxel_size):
    # Load data
    if not os.path.exists(stem_path) or not os.path.exists(canopy_path):
        print(f"Missing point cloud for {id}: stem={os.path.exists(stem_path)}, canopy={os.path.exists(canopy_path)}")
        print(f"Stem path: {stem_path}"
              f"\nCanopy path: {canopy_path}")
        return pd.DataFrame(), [], []
    stem_pc = load_xyz(stem_path)
    foliage_pc = load_xyz(canopy_path)
    df = pd.read_csv(csv_path)
    df["dbh (cm)"] = df[["DBH_cm_hlayer_0.1", "DBH_cm_hlayer_0.05", "DBH_cm_hlayer_0.5"]].min(axis=1)
    bm_stem = np.zeros(len(df))
    bm_branch = np.zeros(len(df))
    for sp in ["Pine", "Spruce", "Birch"]:
        mask = df["Species"] == sp
        stem, branch = MarklundBiomass(
            df.loc[mask, "dbh (cm)"].values * 10,
            sp,
            height_m=df.loc[mask, "Height_m"].values
        )

        if mask.sum() == 0:
            continue

        bm_stem[mask] = stem
        bm_branch[mask] = branch
    df["Biomass_stem"] = bm_stem
    df["Biomass_branch"] = bm_branch


    biomass_stem = df["Biomass_stem"].values[0]
    biomass_foliage = df["Biomass_branch"].values[0]
    #dbh = df["DBH_cm_hlayer_0.5"].values[0]
    #pc_count = df["Point Count"].values[0]

    # Heights
    stem_top = np.max(stem_pc[:, 2]) if len(stem_pc) else 0
    foliage_top = np.max(foliage_pc[:, 2]) if len(foliage_pc) else 0
    total_bottom = min(
        np.min(stem_pc[:, 2]) if len(stem_pc) else np.inf,
        np.min(foliage_pc[:, 2]) if len(foliage_pc) else np.inf
    )

    # Coverage
    coverage_total = (stem_top - total_bottom) / (foliage_top - total_bottom)
    coverage_top = (stem_top) / (foliage_top )
    print(f"Stem coverage: {coverage_total:.2f}")

    #stem_biomass_new = coverage_total * biomass_stem
    #branch_biomass_new = biomass_foliage + (1 - coverage_total) * biomass_stem
    # -----------------------------
    # VOXELIZE
    # -----------------------------
    stem_voxels = voxelize(stem_pc, voxel_size)
    foliage_voxels = voxelize(foliage_pc, voxel_size)
    stem_centers = compute_stem_centers(
        stem_voxels,
        voxel_size
    )
    # -----------------------------
    # DISTRIBUTE BIOMASS
    # -----------------------------
    stem_biomass_vox = distribute_biomass(stem_voxels, biomass_stem)
    foliage_biomass_vox = distribute_biomass(foliage_voxels, biomass_foliage)

    stem_voxel_df = voxels_to_dataframe(
    stem_biomass_vox,
        voxel_size,
        id,
        "stem",
        stem_centers
    )

    foliage_voxel_df = voxels_to_dataframe(
        foliage_biomass_vox,
        voxel_size,
        id,
        "foliage",
        stem_centers
    )

    voxel_df = pd.concat(
        [stem_voxel_df, foliage_voxel_df],
        ignore_index=True
    )

    # -----------------------------
    # AGGREGATE HEIGHT BINS
    # -----------------------------
    stem_layers, stem_bins = aggregate_height_bins(stem_biomass_vox, voxel_size)
    foliage_layers, foliage_bins = aggregate_height_bins(foliage_biomass_vox, voxel_size)

    total_bins = stem_bins + foliage_bins


    # -----------------------------
    # OUTPUT
    # -----------------------------
    output = pd.DataFrame([{
        "id": id[4:],
        "voxel_size": voxel_size,
        "0-1.3m Total": total_bins[0],
        "1.3-8m Total": total_bins[1],
        "8-14m Total": total_bins[2],
        "14+m Total": total_bins[3],
        "0-1.3m Stem": stem_bins[0],
        "1.3-8m Stem": stem_bins[1],
        "8-14m Stem": stem_bins[2],
        "14+m Stem": stem_bins[3],
        "0-1.3m Foliage": foliage_bins[0],
        "1.3-8m Foliage": foliage_bins[1],
        "8-14m Foliage": foliage_bins[2],
        "14+m Foliage": foliage_bins[3]
        
    }])
    df["TreeID"] = pd.to_numeric(df["TreeID"], errors="coerce")
    output["id"] = pd.to_numeric(output["id"], errors="coerce")

    return df.merge(output, left_on="TreeID", right_on="id", how="left"), stem_layers, foliage_layers, voxel_df


def plot_voxel_slices_filled(stem_layers, foliage_layers, voxel_size, max_slices=6):
    z_values = sorted(stem_layers.keys())

    step = max(1, len(z_values) // max_slices)
    z_values = z_values[::step]

    fig, axes = plt.subplots(1, len(z_values), figsize=(4*len(z_values), 4))

    if len(z_values) == 1:
        axes = [axes]

    for ax, z in zip(axes, z_values):
        print(f"z={z:.2f}, stem={len(stem_layers.get(z, []))}, foliage={len(foliage_layers.get(z, []))}")
        # STEM
        for (x, y) in stem_layers.get(z, []):
            rect = patches.Rectangle(
                (x - voxel_size/2, y - voxel_size/2),
                voxel_size,
                voxel_size,
                alpha=0.8
            )
            ax.add_patch(rect)

        # FOLIAGE
        for (x, y) in foliage_layers.get(z, []):
            rect = patches.Rectangle(
                (x - voxel_size/2, y - voxel_size/2),
                voxel_size,
                voxel_size,
                alpha=0.4
            )
            ax.add_patch(rect)
        all_xy = []

        for pts in stem_layers.values():
            all_xy.extend(pts)
        for pts in foliage_layers.values():
            all_xy.extend(pts)

        all_xy = np.array(all_xy)

        xmin, ymin = all_xy.min(axis=0)
        xmax, ymax = all_xy.max(axis=0)

        pad = voxel_size * 2

        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_title(f"z = {z:.2f} m")
        ax.set_aspect("equal")
        ax.grid()

    plt.suptitle(f"Voxel slices (voxel size = {voxel_size} m)")
    plt.savefig(f"/mnt/c/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/voxel_slices_{voxel_size}.png", dpi=311)
    plt.close()


def plot_voxel_distribution(all_stem_layers, all_foliage_layers, aggregate_bins=False):
    def voxel_counts_per_z(layers, voxel_size):
        z_sorted = sorted(layers.keys())
        z = np.array(z_sorted)
        counts = np.array([len(layers[zi]) for zi in z_sorted]) * (voxel_size**3) 
        return z, counts
    def voxel_window_count(layers, z_center, voxel_size, window=1.0):
        return sum(
            len(voxels) * (voxel_size**3) 
            for z, voxels in layers.items()
            if abs(z - z_center) <= window
        )
    plt.figure(figsize=(8, 6))
    if not aggregate_bins:
        suffix = "_full"
        for voxel_size in sorted(all_stem_layers.keys()):

            stem_layers = all_stem_layers[voxel_size]
            foliage_layers = all_foliage_layers[voxel_size]

            z_s, c_s = voxel_counts_per_z(stem_layers, voxel_size)
            z_f, c_f = voxel_counts_per_z(foliage_layers, voxel_size)

            plt.plot(z_s, c_s, label=f"Stem {voxel_size} m")
            plt.plot(z_f, c_f, linestyle="--", label=f"Foliage {voxel_size} m")

        plt.xlabel("Height (z)")
        plt.ylabel("Total voxel volume (±1 m window) (m³)")
        plt.title("Voxel density vs height (full profile)")
    else:
        suffix = "_agg"
        z_targets = [1.3, 8.0, 14.0]
        labels = ["1.3 m", "8 m", "14 m"]

        x = np.arange(len(z_targets))
        width = 0.25
        ratios = {
            "1.3/8": [],
            "8/14": [],
            "1.3/14": []
        }

        voxel_sizes = []
        for i, voxel_size in enumerate(sorted(all_stem_layers.keys())):

            foliage_layers = all_foliage_layers[voxel_size]
            foliage_vals = [voxel_window_count(foliage_layers, z, voxel_size) for z in z_targets]
            voxel_sizes.append(voxel_size)
            offset = (i - 1) * width
            plt.bar(x + offset, foliage_vals, width, alpha=0.5, label=f"Foliage {voxel_size} m")

        plt.xticks(x, labels)
        plt.ylabel("Total voxel volume (±1 m window) (m³)")
        plt.title("Total voxel volume at key heights")

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"/mnt/c/Users/digit/Downloads/Examensarbete/Results/per_voxel_distribution{suffix}.png", dpi=311)
    plt.close()

if __name__ == "__main__":
    input_path = r"/mnt/c/Users/digit/Downloads/Examensarbete/Results/saptrees_segment/"
    tree_list = [file.strip("_").split("_")[0] for file in os.listdir(input_path) if file.endswith("_results.csv")]
    all_stem_layers = {}
    all_foliage_layers = {}
    all_voxel_data = []
    for voxel_size in VOXEL_SIZES:
        biomass_data = pd.DataFrame()
        for id in tqdm(tree_list):
            print(f"Processing tree {id}...")
            biomass_dist, stem_layers, foliage_layers, voxel_df = process_tree(
                stem_path=f"/mnt/c/Users/digit/Downloads/Examensarbete/Results/saptrees_segment/pointclouds/stem/{id}_stempoints.ply",
                canopy_path=f"/mnt/c/Users/digit/Downloads/Examensarbete/Results/saptrees_segment/pointclouds/canopy/{id}_canopypoints.ply",
                csv_path=f"/mnt/c/Users/digit/Downloads/Examensarbete/Results/saptrees_segment/{id}_results.csv",
                id = id,
                voxel_size = voxel_size
            )
            all_stem_layers[voxel_size] = stem_layers
            all_foliage_layers[voxel_size] = foliage_layers
            all_voxel_data.append(voxel_df)
            #plot_voxel_slices_filled(stem_layers, foliage_layers, voxel_size)

            biomass_data = pd.concat([biomass_data, biomass_dist], ignore_index=True)

        biomass_data.to_csv(f"/mnt/c/Users/digit/Downloads/Examensarbete/Results/saptrees_segment/biomass_height_distribution_summary_{voxel_size}.csv", index=False)
    #plot_voxel_distribution(all_stem_layers, all_foliage_layers, aggregate_bins=True)
    all_voxel_data = pd.concat(all_voxel_data, ignore_index=True)
    print(all_voxel_data["direction"].unique())
    all_voxel_data.to_parquet(
        f"/mnt/c/Users/digit/Downloads/Examensarbete/Results/saptrees_segment/voxel_biomass_saptrees.parquet",
        index=False
    )


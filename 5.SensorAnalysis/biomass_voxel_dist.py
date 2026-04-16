import numpy as np
import pandas as pd
import open3d as o3d
import os
from tqdm import tqdm


#Requires biomass_venv!!

VOXEL_SIZE = 0.2  # meters
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
        biomass_per_voxel[voxel] = (count / total_points) * total_biomass #Är det verkligen såhär jag vill distribuera biomassa?

    return biomass_per_voxel


def aggregate_height_bins(biomass_voxels, voxel_size):
    bins = np.zeros(len(HEIGHT_BINS))

    for (ix, iy, iz), biomass in biomass_voxels.items():
        # use voxel center (better!)
        z = (iz + 0.5) * voxel_size

        for i, (zmin, zmax) in enumerate(HEIGHT_BINS):
            if zmin <= z < zmax:
                bins[i] += biomass
                break

    return bins


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------
def process_tree(stem_path, canopy_path, csv_path, id):
    # Load data
    if not os.path.exists(stem_path) or not os.path.exists(canopy_path):
        print(f"Missing point cloud for {id}: stem={os.path.exists(stem_path)}, canopy={os.path.exists(canopy_path)}")
        return
    stem = load_xyz(stem_path)
    foliage = load_xyz(canopy_path)
    df = pd.read_csv(csv_path)

    biomass_stem = df["Biomass_stem"].values[0]
    biomass_foliage = df["Biomass_branch"].values[0]
    dbh = df["DBH_cm"].values[0]
    pc_count = df["Point Count"].values[0]

    # Heights
    stem_top = np.max(stem[:, 2]) if len(stem) else 0
    foliage_top = np.max(foliage[:, 2]) if len(foliage) else 0
    total_bottom = min(
        np.min(stem[:, 2]) if len(stem) else np.inf,
        np.min(foliage[:, 2]) if len(foliage) else np.inf
    )

    # Coverage
    coverage_total = (stem_top - total_bottom) / (foliage_top - total_bottom)
    coverage_top = (stem_top) / (foliage_top )
    print(f"Stem coverage: {coverage_total:.2f}")

    if coverage_total < COVERAGE_THRESHOLD:
        print("Coverage below threshold → skipping voxel distribution")
        total_bins = np.zeros(len(HEIGHT_BINS))
    else:
        # -----------------------------
        # VOXELIZE
        # -----------------------------
        stem_voxels = voxelize(stem, VOXEL_SIZE)
        foliage_voxels = voxelize(foliage, VOXEL_SIZE)

        # -----------------------------
        # DISTRIBUTE BIOMASS
        # -----------------------------
        stem_biomass_vox = distribute_biomass(stem_voxels, biomass_stem)
        foliage_biomass_vox = distribute_biomass(foliage_voxels, biomass_foliage)

        # -----------------------------
        # AGGREGATE HEIGHT BINS
        # -----------------------------
        stem_bins = aggregate_height_bins(stem_biomass_vox, VOXEL_SIZE)
        foliage_bins = aggregate_height_bins(foliage_biomass_vox, VOXEL_SIZE)

        total_bins = stem_bins + foliage_bins


    # -----------------------------
    # OUTPUT
    # -----------------------------
    output = pd.DataFrame([{
        "id": id[4:],
        "dbh (cm)": dbh,
        "height": foliage_top,
        "biomass_stem": biomass_stem,
        "biomass_foliage": biomass_foliage,
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
        "14+m Foliage": foliage_bins[3],
        "coverage_norm": coverage_total,
        "coverage_top": coverage_top,
        "Point count": pc_count
        
    }])

    return output


if __name__ == "__main__":
    input_path = r"/mnt/c/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/yrt_new/"
    tree_list = [file.strip("_").split("_")[0] for file in os.listdir(input_path) if file.endswith("_results.csv")]
    biomass_data = pd.DataFrame()
    for id in ["tree1011", "tree1121", "tree1314", "tree1440", "tree879"]: #tqdm(tree_list):
        print(f"Processing tree {id}...")
        biomass_dist = process_tree(
            stem_path=f"/mnt/c/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/yrt_new/pointclouds/stem/{id}_stempoints.ply",
            canopy_path=f"/mnt/c/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/yrt_new/pointclouds/canopy/{id}_canopypoints.ply",
            csv_path=f"/mnt/c/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/yrt_new/{id}_results.csv",
            id = id
        )
        biomass_data = pd.concat([biomass_data, biomass_dist], ignore_index=True)
        #print(biomass_data)
    biomass_data.to_csv("/mnt/c/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/yrt_new/biomass_height_distribution_summary_bugfix.csv", index=False)
    print("Done!")

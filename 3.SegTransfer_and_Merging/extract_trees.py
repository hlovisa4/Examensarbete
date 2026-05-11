#!/usr/bin/env python3
"""
Split a LAS/LAZ point cloud into one file per instance, using an extra dimension
named 'instance_pred'.
"""
from pathlib import Path
import numpy as np
import laspy
import geopandas as gpd
from tqdm import tqdm

def split_by_instance(
    in_path: Path,
    out_dir: Path,
    instance_field: str = "treeID",
    skip_values: list[int] | None = None,
    min_points: int = 1,
    id_list: list[int] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    las = laspy.read(str(in_path))

    # Check that the attribute exists
    dims = set(las.point_format.dimension_names)
    if instance_field not in dims:
        raise ValueError(
            f"'{instance_field}' not found in point dimensions.\n"
            f"Available dimensions: {sorted(dims)}"
        )

    inst = np.asarray(getattr(las, instance_field))
    if inst.ndim != 1:
        inst = inst.reshape(-1)

    scores = None

    # Determine which instance IDs to process
    unique_ids = np.unique(inst)
    
    if skip_values:
        skip_set = set(int(v) for v in skip_values)
        unique_ids = np.array([i for i in unique_ids if int(i) not in skip_set], dtype=unique_ids.dtype)


    if id_list is not None:
        id_set = id_list
    else:
        id_set = unique_ids
    # Write one file per instance
    for iid in tqdm(id_set, desc="Processing instances"):
        print(iid)
        mask = inst == iid
        n = int(mask.sum())
        if n < min_points:
            continue

        sub = las[mask]  # subsets while preserving point format + extra dims

        if scores is not None:
            inst_scores = scores[mask]
            inst_scores = inst_scores[np.isfinite(inst_scores)]
            if inst_scores.size > 0:
                score_val = float(inst_scores.mean())
            else:
                score_val = float("nan")

            score_str = f"__score_{int(score_val)}"
        else:
            score_str = ""

        suffix = ".las"
        out_path = (
        out_dir
        / f"{in_path.stem}_i_{int(iid)}_n{n}{score_str}{suffix}")
        sub.write(str(out_path))



def main():
    source = "ff3d" # or "FF3D"
    if source == "lidr":
        print("Starting split_by_instance for lidR data")
        match_list = gpd.read_file("C:/Users/digit/Downloads/Examensarbete/Results/ff3d_matched_trees_new.gpkg")
        unique_ids = match_list["chm_id"].unique()
        rng = np.random.default_rng(seed=42)
        filtered_ids = unique_ids[unique_ids != 0]
        sampled_ids = rng.choice(filtered_ids, size=100, replace=False)
        print("starting split_by_instance with", len(sampled_ids), "IDs")
        split_by_instance(
            in_path=Path("C:/Users/digit/Downloads/Examensarbete/Results/TLS_labeled_from_ALS_lidr_2.las"),
            out_dir=Path("C:/Users/digit/Downloads/Examensarbete/Results/extracted_trees_lidr_2"),
            instance_field="treeID",
            skip_values=[],
            min_points=30,
            id_list = sampled_ids.tolist(),
        )
    else:
        print("Starting split_by_instance for ff3d data")
        split_by_instance(
           in_path=Path("C:/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/TLS_labeled_from_ALS_ff3d_2.las"),
            out_dir=Path("C:/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/extracted_trees_ff3d_2"),
            instance_field="instance_pred",
            skip_values=[],
            min_points=30,
        )

main()

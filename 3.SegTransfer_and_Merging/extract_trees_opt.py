#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import laspy
from tqdm import tqdm
import geopandas as gpd


def split_by_instance(
    in_path: Path,
    out_dir: Path,
    instance_field: str = "treeID",
    skip_values: set[int] | None = None,
    min_points: int = 1,
    id_list: list[int] | None = None,
) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)

    las = laspy.read(str(in_path))

    if instance_field not in las.point_format.dimension_names:
        raise ValueError(f"{instance_field} not found")

    inst = las[instance_field].astype(np.int32)

    # -----------------------------
    # Build index map ONCE (fast)
    # -----------------------------
    print("Building instance index map...")
    unique_ids, inverse = np.unique(inst, return_inverse=True)

    # filter IDs
    if id_list is not None:
        id_set = set(id_list)
        valid = np.array([i in id_set for i in unique_ids])
    else:
        valid = np.ones_like(unique_ids, dtype=bool)

    if skip_values:
        skip_set = set(skip_values)
        valid &= np.array([i not in skip_set for i in unique_ids])
    print(f"Found {len(unique_ids)} unique IDs, {valid.sum()} after filtering")
    # precompute indices per instance (FAST ACCESS)
    id_to_indices = {}
    for idx, uid in enumerate(unique_ids):
        if not valid[idx]:
            continue

        inds = np.where(inverse == idx)[0]
        if inds.size >= min_points:
            id_to_indices[uid] = inds

    # -----------------------------
    # Write files
    # -----------------------------
    base = las.header  # reuse header (fast)

    for iid, inds in tqdm(id_to_indices.items(), desc="Writing instances"):
        pts = las.points[inds]  # MUCH faster than las[mask]

        out = laspy.LasData(base)
        out.points = pts

        out_path = out_dir / f"{in_path.stem}_i_{int(iid)}_n{len(inds)}.las"
        out.write(str(out_path))



def main():
    source = "lidr" # or "FF3D"
    if source == "lidr":
        split_by_instance(
            in_path=Path("C:/Users/digit/Downloads/Examensarbete/Results/chm_segmentation/tls_als_lidr_final.las"),
            out_dir=Path("C:/Users/digit/Downloads/Examensarbete/Results/chm_segmentation/ALS_TLS_extracted_trees_lidr"),
            instance_field="treeID",
            skip_values=[],
            min_points=30,
        )
    else:
        print("Starting split_by_instance for ff3d data")
        split_by_instance(
           in_path=Path("C:/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/tls_als_ff3d_final.las"),
            out_dir=Path("C:/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/ALS_TLS_extracted_trees_ff3d"),
            instance_field="instance_pred",
            skip_values=[],
            min_points=30,
        )

main()

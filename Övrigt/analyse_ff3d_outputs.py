import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import laspy
from plyfile import PlyData


# ----------------------------
# Config
# ----------------------------
filename = "731200_7133950fixedname"
workspace = Path(r"C:/Users/digit/Downloads/Examensarbete/Results/test_it3/")
iterations = 3

# Which attribute name to use for instances
INSTANCE_DIM = "instance_pred"
INSTANCE_DIM_BY_KIND = {
    "bluepoints": "treeID",      # <-- only BP differs
    # everything else uses instance_pred
}
# ----------------------------
# Helpers
# ----------------------------
def read_instance_labels_from_ply(ply_path: Path, instance_dim: str = INSTANCE_DIM) -> np.ndarray:
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"].data

    # v.dtype.names are the per-vertex fields
    if instance_dim not in v.dtype.names:
        raise KeyError(f"PLY missing '{instance_dim}'. Available: {list(v.dtype.names)}")

    labels = np.asarray(v[instance_dim])
    return labels


def read_instance_labels_from_las(las_path: Path, instance_dim: str = INSTANCE_DIM) -> np.ndarray:
    las = laspy.read(str(las_path))
    if instance_dim not in las.point_format.dimension_names:
        raise KeyError(
            f"LAS missing '{instance_dim}'. Available: {list(las.point_format.dimension_names)}"
        )
    labels = np.asarray(getattr(las, instance_dim))
    return labels


def counts_per_instance(labels: np.ndarray, ignore_negative: bool = False) -> pd.Series:
    labels = np.asarray(labels)
    if ignore_negative:
        labels = labels[labels >= 0]

    if labels.size == 0:
        return pd.Series(dtype="int64")

    uniq, cnt = np.unique(labels, return_counts=True)
    return pd.Series(cnt, index=uniq).sort_index()


# ----------------------------
# Collect stats over iterations
# ----------------------------
rows = []
per_file_instance_counts = {}  # key: (iteration, kind) -> Series(counts)

for it in range(1, iterations + 1):
    # Match your folder/file naming from the logs:
    #   round_2_noisy_score/<name>_noisysegments.ply
    #   round_2/<name>_round2.las
    #   <name>_bluepoints_2.ply
    #   <name>_2.ply  (merged prediction / final?)
    pc_noise = workspace / f"round_{it}_noisy_score" / f"{filename}_noisysegments.ply"
    pc_noise_removes = workspace / f"round_{it}_after_remove_noise_200" / f"{filename}_round{it}.ply"
    pc_filt  = workspace / f"round_{it}" / f"{filename}_round{it}.las"
    pc_filt_ply  = workspace / f"round_{it}" / f"{filename}_round{it}.ply"
    bp       = workspace / f"{filename}_bluepoints_{it}.ply"
    out      = workspace / f"{filename}_{it}.ply"

    files = [
        ("noisysegments", pc_noise, "ply"),
        ("Noise removes", pc_noise_removes, "ply"),
        ("filt_las",     pc_filt,  "las"),
        ("filt_ply",   pc_filt_ply, "ply"),
        ("bluepoints",    bp,       "ply"),
        ("out",           out,      "ply"),
    ]

    for kind, path, ftype in files:
        if not path.exists():
            print(f"[WARN] Missing: {path}")
            continue

        instance_dim = INSTANCE_DIM_BY_KIND.get(kind, INSTANCE_DIM)

        try:
            if ftype == "ply":
                labels = read_instance_labels_from_ply(path, instance_dim)
            else:
                labels = read_instance_labels_from_las(path, instance_dim)

            inst_counts = counts_per_instance(labels, ignore_negative=False)
            per_file_instance_counts[(it, kind)] = inst_counts

            rows.append({
                "iteration": it,
                "kind": kind,
                "path": str(path),
                "n_points_total": int(len(labels)),
                "n_points_labeled": int((labels >= 0).sum()),
                "n_instances": int(inst_counts.shape[0]),
                "mean_pts_per_instance": float(inst_counts.mean()) if len(inst_counts) else np.nan,
            })

        except Exception as e:
            print(f"[ERROR] Failed reading {path}: {e}")

summary = pd.DataFrame(rows).sort_values(["iteration", "kind"])
print(summary)

# Save summary table
summary_csv = workspace / f"{filename}_instance_summary.csv"
summary.to_csv(summary_csv, index=False)
print(f"\nSaved: {summary_csv}")


# Also save per-instance counts (long format) for deeper analysis
long_rows = []
for (it, kind), s in per_file_instance_counts.items():
    for inst_id, npts in s.items():
        long_rows.append({
            "iteration": it,
            "kind": kind,
            "instance_id": int(inst_id),
            "n_points": int(npts),
        })

counts_long = pd.DataFrame(long_rows).sort_values(["iteration", "kind", "instance_id"])
counts_long_csv = workspace / f"{filename}_instance_counts_long.csv"
counts_long.to_csv(counts_long_csv, index=False)
print(f"Saved: {counts_long_csv}")


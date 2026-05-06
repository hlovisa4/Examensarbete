import numpy as np
import laspy
from scipy.spatial import cKDTree
from tqdm import tqdm


def weighted_mode_vectorized(neighbors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Vectorized weighted mode over the k-nearest neighbours for every point.

    neighbors : (N, k) integer array of labels
    weights   : (N, k) float array of inverse-distance weights
    returns   : (N,)  integer array of the weighted-majority label per point
    """
    N, k = neighbors.shape

    # Flatten to 1-D so we can use np.add.at
    flat_labels   = neighbors.ravel()          # (N*k,)
    flat_weights  = weights.ravel()            # (N*k,)
    point_idx     = np.repeat(np.arange(N), k) # row each element belongs to

    # Map raw labels -> compact 0-based indices so we can index an array
    uniq_labels, compact = np.unique(flat_labels, return_inverse=True)
    n_labels = len(uniq_labels)

    # Accumulate weights into a (N, n_labels) score matrix
    scores = np.zeros((N, n_labels), dtype=np.float64)
    np.add.at(scores, (point_idx, compact), flat_weights)

    # Winner per row
    best_compact = scores.argmax(axis=1)       # (N,)
    return uniq_labels[best_compact]           # back to original label space


def stream_transfer_fixed(
    als_path, tls_path, out_path, dist_path,
    chunk_size=500_000, k=7, max_dist=1.7,
    semantic_dim="semantic_pred", instance_dim="instance_pred",
    unknown_label=-2, eps=1e-6, sem=False, max_dist_strict=0.5
):
    als = laspy.read(als_path)
    dist_lst = []

    # ------------------------------------------------------------------ #
    # Build ALS KD-tree
    # ------------------------------------------------------------------ #
    als_xyz = np.column_stack((
        np.asarray(als.x, np.float64),
        np.asarray(als.y, np.float64),
        np.asarray(als.z, np.float64),
    ))
    tree = cKDTree(als_xyz)

    if semantic_dim not in als.point_format.dimension_names:
        sem = False
        print(f"ALS missing '{semantic_dim}'")
    if instance_dim not in als.point_format.dimension_names:
        raise ValueError(f"ALS missing '{instance_dim}'")

    als_sem = np.asarray(getattr(als, semantic_dim)) if sem else None
    als_ins = np.asarray(getattr(als, instance_dim))
    # ------------------------------------------------------------------ #
    # Centroid tree for fallback assignment
    # ------------------------------------------------------------------ #
    valid_mask  = (als_ins != 0) & (als_ins != -1)
    unique_ids  = np.unique(als_ins[valid_mask])

    # Vectorized centroid computation (no Python loop over IDs)
    sort_order  = np.argsort(als_ins[valid_mask])
    sorted_ids  = als_ins[valid_mask][sort_order]
    sorted_xy   = als_xyz[valid_mask][:, :2][sort_order]

    _, first_occ, counts = np.unique(sorted_ids, return_index=True, return_counts=True)
    # sum x and y per group, then divide by count
    cum_x  = np.cumsum(sorted_xy[:, 0])
    cum_y  = np.cumsum(sorted_xy[:, 1])
    end    = first_occ + counts - 1
    sum_x  = cum_x[end] - np.where(first_occ > 0, cum_x[first_occ - 1], 0.0)
    sum_y  = cum_y[end] - np.where(first_occ > 0, cum_y[first_occ - 1], 0.0)
    centroids    = np.column_stack((sum_x / counts, sum_y / counts))
    centroid_ids = unique_ids  # already same order as np.unique output

    centroid_tree = cKDTree(centroids)

    # ------------------------------------------------------------------ #
    # Stream TLS chunks
    # ------------------------------------------------------------------ #
    with laspy.open(tls_path) as f:
        header = f.header.copy()
        if sem:
            header.add_extra_dim(laspy.ExtraBytesParams(
                name=semantic_dim, type=np.int32))
        header.add_extra_dim(laspy.ExtraBytesParams(
            name=instance_dim, type=np.int32)) #LIDR float64

        with laspy.open(out_path, mode="w", header=header) as writer:
            for points in tqdm(f.chunk_iterator(chunk_size), desc="Processing chunks"):
                pts = np.column_stack((
                    np.asarray(points.x, np.float64),
                    np.asarray(points.y, np.float64),
                    np.asarray(points.z, np.float64),
                ))

                dists, idx = tree.query(pts, k=k, workers=-1)
                dist_lst.append(float(np.mean(dists)))

                # Per-point distance threshold
                nearest_labels = als_ins[idx[:, 0]]
                bad_mask   = (nearest_labels == 0)
                thresholds = np.where(bad_mask, max_dist_strict, max_dist)
                ok         = dists[:, 0] <= thresholds

                # Inverse-distance weights  (N, k)
                w = 1.0 / np.maximum(dists, eps)

                # ---- Vectorized weighted mode ---- #
                ins_neighbors = als_ins[idx]                      # (N, k)
                ins_pred = weighted_mode_vectorized(ins_neighbors, w).astype(np.int32) #LIDR float64

                if sem:
                    sem_neighbors = als_sem[idx]                  # (N, k)
                    sem_pred = weighted_mode_vectorized(sem_neighbors, w).astype(np.int32)
                    sem_pred[~ok] = unknown_label

                # ---- Fallback: centroid tree for failed points ---- #
                failed_xy = pts[~ok, :2]
                if len(failed_xy) > 0:
                    d_c, c_idx = centroid_tree.query(failed_xy, k=1)
                    close = d_c < 3.5

                    # Build a writable index into ~ok rows
                    fail_indices = np.where(~ok)[0]
                    ins_pred[fail_indices[close]]  = centroid_ids[c_idx[close]]
                    ins_pred[fail_indices[~close]] = unknown_label
                else:
                    ins_pred[~ok] = unknown_label

                # ---- Write output ---- #
                out_points = laspy.ScaleAwarePointRecord.zeros(
                    len(points), header=header)
                for dim in points.point_format.dimension_names:
                    if dim in out_points.point_format.dimension_names:
                        out_points[dim] = points[dim]

                if sem:
                    out_points[semantic_dim] = sem_pred
                out_points[instance_dim] = ins_pred
                writer.write_points(out_points)

    with open(dist_path, "w") as f:
        for v in dist_lst:
            f.write(f"{round(v, 3)}\n")

    print("Done:", out_path)


if __name__ == "__main__":
    mode = input("Enter mode (ff3d/lidr): ").strip().lower()
    if mode == "ff3d":
        stream_transfer_fixed(
            als_path=r"C:/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/tile_ALS_clipped_to_reflist.las",
            tls_path=r"C:/Users/digit/Downloads/Examensarbete/Data/TLS_clipped_to_reflist.las",
            out_path=r"C:/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/TLS_labeled_from_ALS_ff3d.las",
            dist_path=r"C:/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/TLS_labeled_from_ALS_ff3d_distances.txt",
            chunk_size=1_000_000,
            k=7,
            max_dist=0.85,
            sem=True,
        )
    elif mode == "lidr":
        stream_transfer_fixed(
            als_path=r"C:/Users/digit/Downloads/Examensarbete/Results/chm_segmentation/chm_ALS_clipped_to_reflist.las",
            tls_path=r"C:/Users/digit/Downloads/Examensarbete/Data/TLS_clipped_to_reflist.las",
            out_path=r"C:/Users/digit/Downloads/Examensarbete/Results/chm_segmentation/TLS_labeled_from_ALS_lidr.las",
            dist_path=r"C:/Users/digit/Downloads/Examensarbete/Results/chm_segmentation/TLS_labeled_from_ALS_lidr_distances.txt",
            chunk_size=1_000_000,
            k=7,
            max_dist=0.85,
            instance_dim="treeID",
        )
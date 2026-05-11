import numpy as np
import laspy
from scipy.spatial import cKDTree
from tqdm import tqdm

def stream_transfer_fixed(
    als_path, tls_path, out_path, dist_path,
    chunk_size=500_000, k=7, max_dist=0.6,
    semantic_dim="semantic_pred", instance_dim="instance_pred",
    unknown_label=-2, eps=1e-6, sem = False, max_dist_strict = 0.5
):
    als = laspy.read(als_path)
    dist_lst = []
    # Build ALS KD-tree
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

    #Get centroids of ALS instances
    valid_mask = (als_ins != 0) & (als_ins != -1)
    unique_ids = np.unique(als_ins[valid_mask])

    centroids = []
    centroid_ids = []

    for uid in unique_ids:
        pts = als_xyz[als_ins == uid]
        xy_centroid = pts[:, :2].mean(axis=0)
        centroids.append(xy_centroid)
        centroid_ids.append(uid)

    centroids = np.array(centroids)
    centroid_ids = np.array(centroid_ids)

    centroid_tree = cKDTree(centroids)

    with laspy.open(tls_path) as f:
        header = f.header.copy()
        if sem:
            header.add_extra_dim(laspy.ExtraBytesParams(
            name=semantic_dim, type=np.int32
                ))
        header.add_extra_dim(laspy.ExtraBytesParams(
                name=instance_dim, type=np.float64
            ))
        
        with laspy.open(out_path, mode="w", header=header) as writer:
            for points in tqdm(f.chunk_iterator(chunk_size), desc="Processing chunks"):
                pts = np.column_stack((
                    np.asarray(points.x, np.float64),
                    np.asarray(points.y, np.float64),
                    np.asarray(points.z, np.float64),
                ))
                dists, idx = tree.query(pts, k=k, workers=-1)
                dist_lst.append(np.mean(dists))
                #ok = dists[:, 0] <= max_dist

                w = 1.0 / np.maximum(dists, eps)
                sem_neighbors = als_sem[idx] if sem else None ## lista med alla nn's seg labels 
                ins_neighbors = als_ins[idx]
                nearest_labels = ins_neighbors[:, 0]
                bad_mask = (nearest_labels == 0)    # | (nearest_labels == -1)
                thresholds = np.where(bad_mask, max_dist_strict, max_dist)
                ok = dists[:, 0] <= thresholds

                if sem:
                    sem_pred = np.empty(len(points), dtype=np.int32)
                ins_pred = np.empty(len(points), dtype=np.int64)

                for i in range(len(points)):
                    # semantic weighted mode
                    ws = w[i]
                    if sem:
                        labs = sem_neighbors[i]
                        uniq, inv = np.unique(labs, return_inverse=True)
                        sem_pred[i] = uniq[np.argmax(np.bincount(inv, weights=ws))]  ####

                    # instance weighted mode
                    labs2 = ins_neighbors[i]
                    uniq2, inv2 = np.unique(labs2, return_inverse=True)
                    ins_pred[i] = uniq2[np.argmax(np.bincount(inv2, weights=ws))]
                    

                if sem:
                    sem_pred[~ok] = unknown_label
                failed_pts_xy = pts[~ok, :2]
                if len(failed_pts_xy) > 0:
                    d, centroid_idx = centroid_tree.query(failed_pts_xy, k=1)
                    valid = d < max_dist
                    ins_pred[~ok][valid] = centroid_ids[centroid_idx[valid]]
                    ins_pred[~ok][~valid] = unknown_label
                    #ins_pred[~ok] = centroid_ids[centroid_idx]
                else:
                    print("No failed points in this chunk.")
                    ins_pred[~ok] = unknown_label
                out_points = laspy.ScaleAwarePointRecord.zeros(len(points), header = header)

                # copy all “standard” dims that exist in the input point format
                for dim in points.point_format.dimension_names:
                    # avoid copying dims that are extras only in output
                    if dim in out_points.point_format.dimension_names:
                        out_points[dim] = points[dim]

                # now set extras
                if sem:
                    out_points[semantic_dim] = sem_pred
                out_points[instance_dim] = ins_pred 
                writer.write_points(out_points)

    rounded_data = [round(float(d), 3) for d in dist_lst]
    with open(dist_path, 'w') as f:
        for value in rounded_data:
            f.write(f"{value}\n")

    print("Done:", out_path)

if __name__ == "__main__":
    mode = input("Enter mode (ff3d/lidr): ").strip().lower()
    if  mode == "ff3d":
        stream_transfer_fixed(
            als_path=r"C:/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/remerged_ff3d_segmented_cloud_plus_missing_points_clipped.las",
            tls_path=r"C:/Users/digit/Downloads/Examensarbete/Data/radarTower001_clipped.las",
            out_path=r"C:/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/TLS_labeled_from_ALS_ff3d_2.las",
            dist_path = r"C:/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/TLS_labeled_from_ALS_ff3d_distances2.txt",
            chunk_size=1_000_000,  # start smaller on Windows
            k=7,
            max_dist=2,
            sem = True
        )
    elif mode == "lidr":
        stream_transfer_fixed(
            als_path=r"C:/Users/digit/Downloads/Examensarbete/Results/ALS_lidr_segmentation.las",
            tls_path=r"C:/Users/digit/Downloads/Examensarbete/Data/radarTower001_clipped.las",
            out_path=r"C:/Users/digit/Downloads/Examensarbete/Results/TLS_labeled_from_ALS_lidr_2.las",
            dist_path = r"C:/Users/digit/Downloads/Examensarbete/Results/TLS_labeled_from_ALS_lidr_distances.txt",
            chunk_size=500_000,  # start smaller on Windows
            k=7,
            max_dist=2,
            instance_dim="treeID"
        )

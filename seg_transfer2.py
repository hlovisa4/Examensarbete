import numpy as np
import laspy
from scipy.spatial import cKDTree

def stream_transfer_fixed(
    als_path, tls_path, out_path,
    chunk_size=500_000, k=7, max_dist=0.6,
    semantic_dim="semantic_pred", instance_dim="instance_pred",
    unknown_label=-2, eps=1e-6
):
    als = laspy.read(als_path)
    dist_lst = []
    # Build ALS KD-tree
    als_xyz = np.column_stack((
        np.asarray(als.x, np.float64),
        np.asarray(als.y, np.float64),
        np.asarray(als.z, np.float64),
    ))
    def extent_str(xyz):
        mn = xyz.min(axis=0)
        mx = xyz.max(axis=0)
        return f"min={mn}, max={mx}, span={mx-mn}"


    tree = cKDTree(als_xyz)

    #if semantic_dim not in als.point_format.dimension_names:
        #raise ValueError(f"ALS missing '{semantic_dim}'")
    if instance_dim not in als.point_format.dimension_names:
        raise ValueError(f"ALS missing '{instance_dim}'")

    als_sem = np.asarray(getattr(als, semantic_dim))
    als_ins = np.asarray(getattr(als, instance_dim))


    with laspy.open(tls_path) as f:
        header = f.header.copy()

        # 🔹 Add extra dims if they don't exist
        header.add_extra_dim(laspy.ExtraBytesParams(
           name=semantic_dim, type=np.int32
            ))

        header.add_extra_dim(laspy.ExtraBytesParams(
                name=instance_dim, type=np.int32
            ))
        with laspy.open(out_path, mode="w", header=header) as writer:
            it = 0
            for points in f.chunk_iterator(chunk_size):
                pts = np.column_stack((
                    np.asarray(points.x, np.float64),
                    np.asarray(points.y, np.float64),
                    np.asarray(points.z, np.float64),
                ))
                dists, idx = tree.query(pts, k=k, workers=-1)
                print(dists[0])
                dist_lst.append(np.mean(dists))
                ok = dists[:, 0] <= max_dist
                print(sum(ok))

                w = 1.0 / np.maximum(dists, eps)

                sem_neighbors = als_sem[idx] ## lista med alla nn's seg labels 
                ins_neighbors = als_ins[idx]

                sem_pred = np.empty(len(points), dtype=np.int32)
                ins_pred = np.empty(len(points), dtype=np.int32)

                for i in range(len(points)):
                    # semantic weighted mode
                    labs = sem_neighbors[i]
                    ws = w[i]
                    uniq, inv = np.unique(labs, return_inverse=True)
                    sem_pred[i] = uniq[np.argmax(np.bincount(inv, weights=ws))]  ####

                    # instance weighted mode
                    labs2 = ins_neighbors[i]
                    uniq2, inv2 = np.unique(labs2, return_inverse=True)
                    ins_pred[i] = uniq2[np.argmax(np.bincount(inv2, weights=ws))]

                sem_pred[~ok] = unknown_label
                ins_pred[~ok] = unknown_label
                out_points = laspy.ScaleAwarePointRecord.zeros(len(points), header = header)

                # copy all “standard” dims that exist in the input point format
                for dim in points.point_format.dimension_names:
                    # avoid copying dims that are extras only in output
                    if dim in out_points.point_format.dimension_names:
                        out_points[dim] = points[dim]

                # now set extras
                out_points[semantic_dim] = sem_pred
                out_points[instance_dim] = ins_pred

                writer.write_points(out_points)

                #print(sem_pred[0:10])
                #print(np.min(points.x), np.max(points.x))
                #setattr(points, semantic_dim, sem_pred)
                #setattr(points, instance_dim, ins_pred)
                #print(np.unique(points.semantic_pred))
               #writer.write_points(points) #Det är här det blir tokigt 
                it += 1
                print(f"Done with iteration {it}")

    print([round(float(d), 3) for d in dist_lst])
    print("Done:", out_path)

if __name__ == "__main__":
    stream_transfer_fixed(
        als_path=r"C:/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/output/731340_7133960fixedname_round1.las",
        tls_path=r"C:/Users/digit/Downloads/Examensarbete/Results/radarTower001_clipped_2.las",
        out_path=r"C:/Users/digit/Downloads/Examensarbete/Results/TLS_labeled_from_ALS_ff3d_prel.las",
        chunk_size=500_000,  # start smaller on Windows
        k=7,
        max_dist=2,
    )

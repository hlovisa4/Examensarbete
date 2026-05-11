import laspy
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

tls_path_1 = r"C:/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/TLS_labeled_from_ALS_ff3d.las"
tls_path_2 = r"C:/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/TLS_labeled_from_ALS_ff3d_2.las"
chunk_size = 1_000_000
instance_dim = "instance_pred"


unique_instances = set()
points_per_instance = defaultdict(lambda: {"file1": 0, "file2": 0})

with laspy.open(tls_path_1) as f:
    # Check that the dimension exists
    if instance_dim not in f.header.point_format.dimension_names:
        raise ValueError("LAS file is missing 'treeID'")

    for points in tqdm(f.chunk_iterator(chunk_size), desc="Processing chunks"):
        # Read the treeID values for this chunk
        inst = points.instance_pred

        # Add unique values from this chunk to the global set
        unique_instances.update(inst.tolist())

        for val in inst:
           points_per_instance[val]["file1"] += 1

with laspy.open(tls_path_2) as f:
    # Check that the dimension exists
    if instance_dim not in f.header.point_format.dimension_names:
        raise ValueError("LAS file is missing 'treeID'")

    for points in tqdm(f.chunk_iterator(chunk_size), desc="Processing chunks"):
        # Read the treeID values for this chunk
        inst = points.instance_pred

        # Add unique values from this chunk to the global set
        unique_instances.update(inst.tolist())

        for val in inst:
            points_per_instance[val]["file2"] += 1
    
    

# Convert to sorted list for nicer printing
unique_instances = sorted(unique_instances)

rows = []
for tree_id, counts in points_per_instance.items():
    rows.append({
        "treeID": tree_id,
        "file1_count": counts["file1"],
        "file2_count": counts["file2"],
        "difference": counts["file2"] - counts["file1"]
    })

df = pd.DataFrame(rows)

df.to_csv("treeID_comparison.csv", index=False)


#print("\nPoints per instance:")
#for inst_id in unique_instances:
#    print(f"Instance {inst_id}: {points_per_instance[inst_id]} points")
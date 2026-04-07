import laspy
from collections import defaultdict

tls_path = r"C:/Users/digit/Downloads/Examensarbete/Results/TLS_labeled_from_ALS_lidr.las"
chunk_size = 1_000_000


unique_instances = set()
points_per_instance = defaultdict(int)

i = 0
with laspy.open(tls_path) as f:
    # Check that the dimension exists
    if "instance_pred" not in f.header.point_format.dimension_names:
        raise ValueError("LAS file is missing 'instance_pred'")

    for points in f.chunk_iterator(chunk_size):
        # Read the instance_pred values for this chunk
        inst = points.instance_pred

        # Add unique values from this chunk to the global set
        unique_instances.update(inst.tolist())

        for val in inst:
            points_per_instance[val] += 1

        i+=1
        print(i)

# Convert to sorted list for nicer printing
unique_instances = sorted(unique_instances)


print("\nPoints per instance:")
for inst_id in unique_instances:
    print(f"Instance {inst_id}: {points_per_instance[inst_id]} points")
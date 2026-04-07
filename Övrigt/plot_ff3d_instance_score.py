from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData

ply_path = Path(r"C:/Users/digit/Downloads/Examensarbete/Results/test_it3/round_3_noisy_score/731200_7133950fixedname_noisysegments.ply")

ply = PlyData.read(str(ply_path))
v = ply["vertex"].data

print("Vertex fields:", v.dtype.names)

field = "instance_score"
if field not in v.dtype.names:
    raise KeyError(f"PLY missing '{field}'. Available: {list(v.dtype.names)}")

scores = np.asarray(v[field])

# Drop NaN/inf just in case
scores = scores[np.isfinite(scores)]

threshold = 1000
scores= scores[scores <= threshold]

print(f"Count: {scores.size}")
print(f"Min: {scores.min():.3f}  Max: {scores.max():.3f}")
print(f"Mean: {scores.mean():.3f}  Median: {np.median(scores):.3f}")

plt.figure()
plt.hist(scores, bins=50)  # change bins if you want
plt.xlabel("instance_score")
plt.ylabel("Number of points")
plt.title("Histogram of instance_score")
plt.tight_layout()
plt.show()

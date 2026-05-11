import laspy
import numpy as np
import geopandas as gpd
from shapely.geometry import MultiPoint
import shapely.vectorized as sv

# -------------------
# 1. Load TLS
# -------------------
tls_path = "C:/Users/digit/Downloads/Examensarbete/Results/TLS_labeled_from_ALS_lidr_260304_clipped_thin.las"
tls = laspy.read(tls_path)

# Extract XY
xy_tls = np.vstack((tls.x, tls.y)).T

# -------------------
# 2. Convex hull + buffer
# -------------------
polygon = MultiPoint(xy_tls).convex_hull

# Add 1 meter buffer (IMPORTANT)
polygon = polygon.buffer(1.0)

# -------------------
# 3. Save polygon (optional)
# -------------------
gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:3006")
gdf.to_file("tls_convex_hull_buffered.gpkg", driver="GPKG")
print("Polygon saved as tls_convex_hull_buffered.gpkg")

# -------------------
# 4. Load ALS
# -------------------
als_path = "C:/Users/digit/Downloads/Examensarbete/Results/ALS_lidr_segmentation.las"
als = laspy.read(als_path)

# -------------------
# 5. Clip ALS (FAST vectorized)
# -------------------
mask = sv.contains(polygon, als.x, als.y)

# -------------------
# 6. Save result
# -------------------
out = laspy.LasData(als.header)
out.points = als.points[mask]

out_path = "C:/Users/digit/Downloads/Examensarbete/Results/ALS_lidr_segmentation_clipped.las"
out.write(out_path)

print("Done: ALS clipped with convex hull + 1m buffer")
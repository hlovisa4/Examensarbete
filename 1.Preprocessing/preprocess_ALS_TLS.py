import laspy
import numpy as np
import geopandas as gpd
from shapely.geometry import MultiPoint
import shapely.vectorized as sv


# Read gpkg
gdf = gpd.read_file("your_file.gpkg")

# Reproject to a CRS with meters (important!)
gdf = gdf.to_crs(3006)  # SWEREF99 TM (Sweden)

# Create convex hull
hull = gdf.unary_union.convex_hull

# Buffer 10 meters
poly_10m = gpd.GeoDataFrame(geometry=[hull.buffer(10)], crs=gdf.crs)

# Save
poly_10m.to_file("output_polygon.gpkg", driver="GPKG")



###ALS 
als_path = "C:/Users/digit/Downloads/Examensarbete/Data"
als = laspy.read(als_path)

mask = sv.contains(poly_10m, als.x, als.y)

out = laspy.LasData(als.header)
out.points = als.points[mask]
out_path = "C:/Users/digit/Downloads/Examensarbete/Data/ALS_clipped.las"
out.write(out_path)
print("Done: ALS clipped with convex hull + 1m buffer")


###TLS 
tls_path = "C:/Users/digit/Downloads/Examensarbete/Data/"
tls = laspy.read(tls_path)

mask = sv.contains(poly_10m, tls.x, tls.y)

out = laspy.LasData(tls.header)
out.points = tls.points[mask]
out_path = "C:/Users/digit/Downloads/Examensarbete/Data/TLS_clipped.las"
out.write(out_path)
print("Done: ALS clipped with convex hull + 1m buffer")
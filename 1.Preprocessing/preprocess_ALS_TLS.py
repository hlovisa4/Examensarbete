import os
import laspy
import numpy as np
import geopandas as gpd
from shapely.geometry import MultiPoint
from shapely.geometry import Polygon
from shapely import contains_xy

# Read gpkg
gdf = gpd.read_file("C:/Users/Lovisa/Downloads/Examensarbete/5.SensorAnalysis/ff3d_matched_trees_with_biomass.gpkg")
gdf = gdf.to_crs(3006)
# Create convex hull
hull = gdf.geometry.union_all().convex_hull

# Buffer 10 meters
poly_10m = gpd.GeoDataFrame(geometry=[hull.buffer(5)], crs=gdf.crs)
poly_10m.to_file("C:/Users/Lovisa/Downloads/Examensarbete/5.SensorAnalysis/reflist_polygon.gpkg", driver="GPKG")
poly_geom = poly_10m.geometry.iloc[0]


###ALS 
als_path = "C:/Users/Lovisa/Downloads/240829_ALS_Matrice300_Svb/240829_ALS_Matrice300_Svb.las"
als = laspy.read(als_path)

mask = contains_xy(poly_geom, np.asarray(als.x), np.asarray(als.y))

out = laspy.LasData(als.header)
out.points = als.points[mask]
out_path = "C:/Users/Lovisa/Downloads/ALS_clipped.las"
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
import os
import laspy
import numpy as np
import geopandas as gpd
from shapely.geometry import MultiPoint
from shapely.geometry import Polygon
from shapely import contains_xy
from tqdm import tqdm

# Read gpkg
gdf = gpd.read_file("C:/Users/digit/Downloads/Examensarbete/Data/TreesTowerFoot240829.gpkg")
gdf = gdf.to_crs(3006)
# Create convex hull
hull = gdf.geometry.union_all().convex_hull

# Buffer 10 meters
poly_10m = gpd.GeoDataFrame(geometry=[hull.buffer(5)], crs=gdf.crs)
#poly_10m.to_file("C:/Users/Lovisa/Downloads/Examensarbete/5.SensorAnalysis/reflist_polygon.gpkg", driver="GPKG")
poly_geom = poly_10m.geometry.iloc[0]


###ALS 
als_path = "C:/Users/digit/Downloads/Examensarbete/Results/chm_segmentation/ALS_lidr_segmentation_clipped.las"
als = laspy.read(als_path)

mask = contains_xy(poly_geom, np.asarray(als.x), np.asarray(als.y))

out = laspy.LasData(als.header)
out.points = als.points[mask]
out_path = "C:/Users/digit/Downloads/Examensarbete/Results/chm_segmentation/chm_ALS_clipped_to_reflist.las"
out.write(out_path)
print("Done: ALS clipped with convex hull + 1m buffer")


###TLS 
tls_path = "C:/Users/digit/Downloads/Examensarbete/Data/gthradarTowerTLS_2023/R1/radarTower001.las"
out_path = "C:/Users/digit/Downloads/Examensarbete/Data/TLS_clipped_to_saptrees.las"

chunk_size =1_000_000
with laspy.open(str(tls_path)) as reader:
    header = reader.header

    with laspy.open(str(out_path), mode="w", header=header) as writer:
        for pts in tqdm(reader.chunk_iterator(chunk_size)):
            # Scale coords to float for accurate spatial checking
            x = np.asarray(pts.x)
            y = np.asarray(pts.y)

            # Check which points are inside the polygon
            mask = contains_xy(poly_geom, x, y)

            if np.any(mask):
                writer.write_points(pts[mask])

print("Done: TLS clipped with convex hull + 1m buffer")
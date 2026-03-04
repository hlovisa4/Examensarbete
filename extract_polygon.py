import geopandas as gpd
from pathlib import Path
import numpy as np
import laspy
from shapely.geometry import Polygon
from shapely import contains_xy  # shapely>=2.0

# Load the shapefile
gdf = gpd.read_file('C:/Users/digit/Downloads/studyarea_buff10/studyarea_buff10.shp')

# Access the geometry of the first polygon
polygon_shape = gdf.geometry.iloc[0]

# Print the coordinates (External Ring)
poly_xy = list(polygon_shape.exterior.coords)

# 1. Paths - Ensure these are correct for your machine
in_las  = Path(r"C:/Users/digit/Downloads/Examensarbete/Data/radarTowerTLS_2023/R1/radarTower001.las")
#in_las  = Path(r"C:/Users/digit/Downloads/Examensarbete/Results/TLS_thin.las")
out_las = Path(r"C:/Users/digit/Downloads/Examensarbete/Results/radarTower001_clipped.las")

# Create output directory if it doesn't exist
out_las.parent.mkdir(parents=True, exist_ok=True)

poly = Polygon(poly_xy)

# 3. Processing the LAS file
chunk_size = 2_000_000  # adjust for your RAM

with laspy.open(str(in_las)) as reader:
    header = reader.header

    with laspy.open(str(out_las), mode="w", header=header) as writer:
        for pts in reader.chunk_iterator(chunk_size):
            # Scale coords to float for accurate spatial checking
            x = np.asarray(pts.x)
            y = np.asarray(pts.y)

            # Check which points are inside the polygon
            mask = contains_xy(poly, x, y)

            if np.any(mask):
                writer.write_points(pts[mask])

print(f"Successfully clipped! File saved at: {out_las}")
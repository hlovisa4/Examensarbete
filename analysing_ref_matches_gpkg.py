import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

gdf = gpd.read_file("C:/Users/digit/Downloads/Examensarbete/Results/ff3d_matched_trees.gpkg")

gdf = gdf.sort_values(["matched_dist"], ascending= True)
is_duplicate = gdf.duplicated(subset=["matched_id"], keep="first")
gdf.loc[is_duplicate & gdf["matched_id"].notna(), "matched"] = 0
print(f"Matched: {sum(gdf["matched"])}")
gdf_matches =gdf[gdf["matched"] == 1]
print(f"average height match: {np.mean(gdf_matches['matched_height'])}")


id_to_species = {1: "Pine", 2: "Spruce", 3: "Birch", 7: "Ädel", 11: "Dead" }
gdf["Species_name"] = gdf["Species"].map(id_to_species)

species_accuracy = gdf.groupby("Species_name")["matched"].mean() * 100

# Sort for a better looking plot
species_accuracy = species_accuracy.sort_values(ascending=False)

print(species_accuracy)

#Check RMSE
distances = gdf_matches['matched_dist']
rmse_location = np.sqrt(np.mean(distances**2))
print(f"Location RMSE: {rmse_location:.3f} meters")

bins = [0, 5, 10, 15, 20, 25, 30, 40, 50]
labels = ['0-5m', '5-10m', '10-15m', '15-20m', '20-25m', '25-30m', '30-40m', '40m+']

# Add a height class column to your main gdf
gdf['height_class'] = pd.cut(gdf['H_TLS'], bins=bins, labels=labels)

# Group by height class to see matching rate (Recall)
height_analysis = gdf.groupby('height_class')['matched'].agg(['count', 'sum', 'mean'])
height_analysis.columns = ['Total_Trees', 'Matched_Trees', 'Recall_Rate']
height_analysis['Recall_Rate'] *= 100
print(height_analysis)

gdf.to_file("C:/Users/digit/Downloads/Examensarbete/Results/segmentation_evaluation/ff3d_matched_trees_nodups.gpkg", layer='trees', driver="GPKG")

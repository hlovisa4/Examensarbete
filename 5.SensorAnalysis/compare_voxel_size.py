import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt



reflist = gpd.read_file("C:/Users/digit/Downloads/Examensarbete/Results/matched_trees.gpkg")

biomass = pd.read_csv("C:/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/biomass_height_distribution_summary.csv")
print("reflist duplicates:", reflist["ff3d_tile_id"].duplicated().sum())
print("biomass duplicates:", biomass["TreeID"].duplicated().sum())
#biomass = biomass.groupby("TreeID").mean(numeric_only=True).reset_index()
height_cols = ["0-1.3m Stem", "1.3-8m Stem", "8-14m Stem", "14+m Stem"]
biomass_long = biomass.melt(
    id_vars=["TreeID", "voxel_size"],
    value_vars=height_cols,
    var_name="height_bin",
    value_name="biomass"
)
biomass_summary = (
    biomass_long
    .groupby(["voxel_size", "height_bin"])
    .agg(mean_biomass=("biomass", "mean"),
         median_biomass=("biomass", "median"))
    .reset_index()
)
print(biomass_summary["voxel_size"].unique())

plt.figure()

for voxel_size, df in biomass_summary.groupby("voxel_size"):
    plt.plot(df["height_bin"], df["mean_biomass"], label=f"{voxel_size} m")

plt.xlabel("Height")
plt.ylabel("Mean biomass")
plt.title("Biomass distribution vs height for different voxel sizes")
plt.legend(title="Voxel size")
plt.grid()

plt.show()
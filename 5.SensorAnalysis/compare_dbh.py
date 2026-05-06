import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt



reflist = gpd.read_file("/mnt/c/Users/digit/Downloads/Examensarbete/Results/matched_trees.gpkg")

biomass = pd.read_csv("/mnt/c/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/biomass_height_distribution_summary_dbhtrial.csv")
print("reflist duplicates:", reflist["ff3d_tile_id"].duplicated().sum())
print("biomass duplicates:", biomass["TreeID"].duplicated().sum())
biomass = biomass.groupby("TreeID").mean(numeric_only=True).reset_index()
for hlayer in ["0.5", "0.1", "0.05"]:
    print(f"Comparing with DBH from hlayer {hlayer}...")
    col = f"DBH_cm_hlayer_{hlayer}"
    biomass_filtered = biomass[(biomass[col].notna()) & (biomass[col] <= 100)]
    reflist_filtered = reflist[reflist["DBH_Field"].notna() ]
    
    merged = reflist_filtered.merge(biomass_filtered, left_on="ff3d_tile_id", right_on="TreeID", how="inner")
    print(len(merged))
    merged["DBH_diff"] = merged[f"DBH_cm_hlayer_{hlayer}"] - merged["DBH_Field"]
    rmse_dbh = np.sqrt(np.mean(merged["DBH_diff"]**2))
    median_error = np.median(merged["DBH_diff"])
    print(f"Median DBH error for hlayer {hlayer}: {median_error:.8f} cm")
    print(f"RMSE for DBH comparison with hlayer {hlayer}: {rmse_dbh:.8f} cm")

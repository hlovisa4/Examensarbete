import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

rmses = {"ff3d_full": [], "ff3d_tile": [], "ff3d_ref": [], "chm": []}
matches = {"ff3d_full": [], "ff3d_tile": [], "ff3d_ref": [], "chm": []}

for d in [1.8]: # np.linspace(0, 6, 100):
    for method in ["ff3d_tile", "chm"]:
        gdf = gpd.read_file("C:/Users/digit/Downloads/Examensarbete/Results/matched_trees.gpkg")
        print(f"Evaluating method: {method}")
        gdf.loc[gdf[f"{method}_dist"] >= d, f"{method}_matched"] = 0
        gdf = gdf.sort_values([f"{method}_dist"], ascending= True)
        is_duplicate = gdf.duplicated(subset=[f"{method}_id"], keep="first")
        gdf.loc[is_duplicate & gdf[f"{method}_id"].notna(), f"{method}_matched"] = 0
        print(f"Matched: {sum(gdf[f"{method}_matched"])}")
        matches[method].append(sum(gdf[f"{method}_matched"]))
        id_to_species = {1: "Pine", 2: "Spruce", 3: "Birch", 7: "Ädel", 11: "Dead" }
        gdf["Species_name"] = gdf["Species"].map(id_to_species)        
        species_analysis = gdf.groupby('Species_name')[f"{method}_matched"].agg(['count', 'sum', 'mean'])
        species_analysis.columns = ['Total_Trees', 'Matched_Trees', 'Recall_Rate']
        species_analysis['Recall_Rate'] *= 100
        print(species_analysis)
        bins = [0, 5, 10, 15, 20, 25, 30, 40, 50]
        labels = ['0-5m', '5-10m', '10-15m', '15-20m', '20-25m', '25-30m', '30-40m', '40m+']
        # Add a height class column to your main gdf
        gdf['height_class'] = pd.cut(gdf['H_TLS'], bins=bins, labels=labels)
        # Group by height class to see matching rate (Recall) 
        height_analysis = gdf.groupby('height_class')[f"{method}_matched"].agg(['count', 'sum', 'mean'])
        height_analysis.columns = ['Total_Trees', 'Matched_Trees', 'Recall_Rate']
        height_analysis['Recall_Rate'] *= 100
        print(height_analysis)



        gdf_matches =gdf[gdf[f"{method}_matched"] == 1]
        print(f"average height match: {np.mean(gdf_matches[f"{method}_height_error"])}, standard deviation: {np.std(gdf_matches[f"{method}_height_error"])}")

        #Check RMSE
        distances = gdf_matches[f"{method}_dist"]
        rmse_location = np.sqrt(np.mean(distances**2))
        rmse_std = np.sqrt(np.var(distances**2))
        dist_med = np.median(distances)
        print(f"Location RMSE: {rmse_location:.3f}, Standard Deviation: {rmse_std:.3f} meters, Median: {dist_med:.3f} meters")
        rmses[method].append(rmse_location)

  
        height_error_analysis = gdf_matches.groupby('height_class')[f"{method}_height_error"].agg(
            ['count', 'mean', 'std', 'median']
        )

        # Optional: RMSE per height class
        height_error_analysis['rmse'] = gdf_matches.groupby('height_class')[f"{method}_height_error"].apply(
            lambda x: np.sqrt(np.mean(x**2))
        )

        print("\nHeight error per height class:")
        print(height_error_analysis)


plt.plot(np.linspace(0, 6, 100), rmses["ff3d_full"], label="ff3d_full")
plt.plot(np.linspace(0, 6, 100), rmses["ff3d_tile"], label="ff3d_tile")
plt.plot(np.linspace(0, 6, 100), rmses["ff3d_ref"], label="ff3d_ref")
plt.plot(np.linspace(0, 6, 100), rmses["chm"], label="chm")
plt.xlabel("Distance Threshold (m)")
plt.ylabel("RMSE (m)")
plt.title("RMSE vs. Distance Threshold")
plt.legend()
plt.show()

plt.plot(np.linspace(0, 6, 100), matches["ff3d_full"], label="ff3d_full")
plt.plot(np.linspace(0, 6, 100), matches["ff3d_tile"], label="ff3d_tile")
plt.plot(np.linspace(0, 6, 100), matches["chm"], label="chm")
plt.xlabel("Distance Threshold (m)")   
plt.ylabel("Number of Matches")
plt.title("Number of Matches vs. Distance Threshold")
plt.legend()
plt.show()
#gdf.to_file("C:/Users/digit/Downloads/Examensarbete/Results/segmentation_evaluation/ff3d_matched_trees_nodups.gpkg", layer='trees', driver="GPKG")

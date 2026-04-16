# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:49:20 2026

@author: Lovisa
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

metadata = pd.read_csv("C:/Users/Lovisa/Downloads/Examensarbete/5.SensorAnalysis/sapflow_sensors_20250925.csv") 
geometry = [Point(xy) for xy in zip(metadata["Easting"], metadata["Northing"])]

#Plocka ut rätt biomassa
ref_gdf = gpd.read_file("C:/Users/Lovisa/Downloads/Examensarbete/5.SensorAnalysis/ff3d_matched_trees_with_biomass.gpkg", driver="GPKG") 

sap_gdf = gpd.GeoDataFrame(
    {"metadata_id": metadata["Name"]}, 
    geometry=geometry, 
    crs=ref_gdf.crs
)

nearest = gpd.sjoin_nearest(sap_gdf, ref_gdf, how="left", distance_col="dist")
id_mapping = {str(k): v for k, v in dict(zip(nearest["metadata_id"], nearest["ff3d_id"])).items()}
print(id_mapping)




BM_df = pd.read_csv("C:/Users/Lovisa/Downloads/Examensarbete/5.SensorAnalysis/biomass_height_distribution_summary.csv")
matched_ids = set(id_mapping.values())
filtered_BM_data = BM_df[BM_df["id"].isin(matched_ids)].copy()
print(f"Matched rows: {len(filtered_BM_data)}")


refs = {
    "47355": {
        1: [403, "Pine", ["Teros12_VWC(1)", "Teros12_VWC(2)", "Teros12_VWC(3)"]],
        2: [408, "Spruce", ["Teros12_VWC(4)", "Teros12_VWC(5)", "Teros12_VWC(6)"]]
    },
    "47356": {
        1: [360, "Spruce", ["Teros12_VWC(1)", "Teros12_VWC(2)", "Teros12_VWC(3)"]], # Added dummy data for consistency
        2: [364, "Pine", ["Teros12_VWC(4)", "Teros12_VWC(5)", "Teros12_VWC(6)"]]
    }
}
height_map = {0: 14, 1: 8, 2: 1.3}

all_data = []

for site_id in refs:
    sap_df = pd.read_csv(f"C:/Users/Lovisa/Downloads/Examensarbete/5.SensorAnalysis/TOA5_{site_id}_Sept22_2025.txt")
    sap_df["TIMESTAMP"] = pd.to_datetime(sap_df["TIMESTAMP"])
    sap_df = sap_df[sap_df["TIMESTAMP"] >= "2025-05-01"] #sortera bort de innan 2025-05-01 eller vad det än är
    current_ref = refs[site_id]
    for tree_num, data in current_ref.items():
        tree_id = str(data[0])
        species = data[1]
        sensor_cols = data[2]
        ff3d_id = id_mapping[tree_id]
        cols = ["0-1.3m", "1.3-8m", "8-14m", "14+m"]
        BM = filtered_BM_data.loc[filtered_BM_data["id"] == ff3d_id, cols]
        if BM.empty:
            print(f"Skipping {tree_id} (ff3d_id: {ff3d_id}): No biomass data found.")
            continue
        for idx, sensor in enumerate(sensor_cols):
            biomass = BM[cols[idx]].iloc[0]
            temp_df = pd.DataFrame({
                "timestamp": sap_df["TIMESTAMP"],
                "WC": sap_df[sensor],
                "tree": tree_id,
                "species": species,
                "height": height_map[idx],
                "sensor_type": sensor,
                "biomass" : biomass
            })
            initial_len = len(temp_df)
            temp_df = temp_df.dropna(subset=["WC"])
            all_data.append(temp_df)
        
output = pd.concat(all_data, ignore_index=True) 
output.to_csv("C:/Users/Lovisa/Downloads/Examensarbete/5.SensorAnalysis/sapanalysis.csv", index=False)
    






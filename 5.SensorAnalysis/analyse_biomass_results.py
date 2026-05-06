import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import geopandas as gpd
import os
from calculate_biomass import MarklundBiomass
from scipy.stats import chi2_contingency
import statsmodels.formula.api as smf

def dbh_success_analysis(df, output_folder):
    """
    Analyses the total success rate for dbh extraction, and compares it to height classes, per species and per dbh class.
    """
    df.loc[df["dbh (cm)"] > 50, "dbh (cm)"] = np.nan
    df["success"] = df["dbh (cm)"].notna() & df["Height_m"].notna()
    dbh_success_rate = len(df[df["success"] == True]) / len(df)
    
    species_accuracy = df.groupby("Species_name")[f"success"].mean() * 100
    species_accuracy = species_accuracy.sort_values(ascending=False)

    df_ps = df[df["Species_name"].isin(["Pine", "Spruce"])].copy()
    df_ps["success_int"] = df_ps["success"].astype(int)
    df_ps["Species_name"] = pd.Categorical(
        df_ps["Species_name"],
        categories=["Pine", "Spruce"]  # pine becomes reference
    )
    contingency = pd.crosstab(df_ps["Species_name"], df_ps["success_int"])

    from scipy.stats import chi2_contingency
    chi2, p, dof, expected = chi2_contingency(contingency)

    print(p)

    height_accuracy = df.groupby("height_class")[f"success"].mean() * 100
    #height_accuracy = height_accuracy.sort_values(ascending=False)

    dbh_accuracy = df.groupby("dbh_class")[f"success"].mean() * 100
    #dbh_accuracy = dbh_accuracy.sort_values(ascending=False)

    plt.figure(figsize=(6, 6))
    plt.scatter(df["Point Count"], df["Stem_coverage"], c = df["success"], alpha=0.5)
    plt.xlabel("Point Count")
    plt.ylabel("Normalized Coverage")
    plt.title("DBH and Height Extraction Success")
    plt.savefig(f"{output_folder}/success_analysis.png", dpi=311)

    print(f"Overall DBH and Height Extraction Success Rate: {dbh_success_rate:.2%}")
    print(f"Accuracy by DBH Class:")
    print(dbh_accuracy)
    print(f"Accuracy by Height Class:")
    print(height_accuracy)
    print(f"Species Accuracy:")
    print(species_accuracy)
    return dbh_success_rate, species_accuracy, height_accuracy, dbh_accuracy


def dbh_analysis(df, output_folder):
    """
    Analyses the DBH estimation.
    Compares estimated DBH to reference DBH, computes bias, relative error, and R², and plots relative error vs. reference DBH
    """
    df.loc[df["dbh (cm)"] > 50, "dbh (cm)"] = np.nan
    df = df.dropna(subset=["dbh (cm)", "dbh_ref (cm)"])
    dbh = df["dbh (cm)"]
    ref_dbh = df["dbh_ref (cm)"]
    bias = np.mean(dbh - ref_dbh)
    bias_std = np.std(dbh - ref_dbh)
    rel_error = np.mean(np.abs(dbh - ref_dbh) / ref_dbh)
    rel_error_std = np.std(np.abs(dbh - ref_dbh) / ref_dbh)
    plt.figure(figsize=(6, 6))
    #plt.scatter(ref_dbh, (dbh - ref_dbh) / ref_dbh,  alpha=0.5)
    plt.scatter(ref_dbh, dbh,  alpha=0.5)
    r2 = r2_score(ref_dbh, dbh)
    plt.text(
        0.05, 0.95,
        f"$R^2 = {r2:.3f}$",
        transform=plt.gca().transAxes,
        verticalalignment='top'
    )
    plt.ylabel("Estimated DBH (cm)")
    plt.xlabel("Reference DBH (cm)")
    plt.savefig(f"{output_folder}/dbh.png", dpi=311)
    print(f"DBH Bias: {bias:.2f} cm, DBH Bias Std: {bias_std:.2f} cm, Relative DBH Error: {rel_error:.2%}, Relative DBH Error Std: {rel_error_std:.2%}, R²: {r2:.3f}")
    return bias, bias_std, rel_error, rel_error_std, r2

def height_analysis(df, output_folder):
    """
    Analyses the height estimation.
    Compares estimated height to reference height, computes bias, relative error, and R², and plots relative error vs. reference height
    """
    df = df.dropna(subset=["height_ref"])
    height = df["Height_m"]
    ref_height = df["height_ref"]
    bias = np.mean(height - ref_height)
    bias_std = np.std(height - ref_height)
    rel_error = np.mean(np.abs(height - ref_height) / ref_height)
    rel_error_std = np.std(np.abs(height - ref_height) / ref_height)
    plt.figure(figsize=(6, 6))
    plt.scatter(ref_height, height, alpha=0.5)
    plt.ylabel("Estimated Height (m)")
    plt.xlabel("Reference Height (m)")
    r2 = r2_score(height, ref_height)
    plt.text(
        0.05, 0.95,
        f"$R^2 = {r2:.3f}$",
        transform=plt.gca().transAxes,
        verticalalignment='top'
    )
    plt.savefig(f"{output_folder}/height.png", dpi=311)

    plt.figure(figsize=(6, 6))
    plt.scatter(df["ff3d_tile_dist"], np.abs(height - ref_height) / ref_height, alpha=0.5)
    r2 = r2_score(df["ff3d_tile_dist"], np.abs(height - ref_height) / ref_height)
    plt.text(
        0.05, 0.95,
        f"$R^2 = {r2:.3f}$",
        transform=plt.gca().transAxes,
        verticalalignment='top'
    )
    plt.ylabel("Relative Height Error")
    plt.xlabel("FF3D Distance (m)")
    plt.savefig(f"{output_folder}/height_error_ff3d.png", dpi=311)

    print(f"Height Bias: {bias:.2f} m, Height Bias Std: {bias_std:.2f} m, Relative Height Error: {rel_error:.2%}, Relative Height Error Std: {rel_error_std:.2%}, R²: {r2:.3f}")
    return bias, bias_std, rel_error, rel_error_std

def plot_dbh_height(df, output_folder):
    """
    Analysing the relationship between DBH and height for both reference and estimated, plotting them against each other, and computing R² for both.
    """
    df = df.dropna(subset=["dbh (cm)", "Height_m", "dbh_ref (cm)", "height_ref"])
    plt.figure(figsize=(6, 6))
    plt.scatter(df["dbh_ref (cm)"], df["height_ref"], label="Reference", alpha=0.5)
    z = np.polyfit(df["dbh_ref (cm)"], df["height_ref"], 1)
    p = np.poly1d(z)
    plt.plot(df["dbh_ref (cm)"], p(df["dbh_ref (cm)"]), label="Reference Trend", color="blue")

    plt.scatter(df["dbh (cm)"], df["Height_m"], label="Estimated", alpha=0.5)
    z = np.polyfit(df["dbh (cm)"], df["Height_m"], 1)
    p = np.poly1d(z)
    plt.plot(df["dbh (cm)"], p(df["dbh (cm)"]), label="Estimated Trend", color="orange")

    plt.xlabel("DBH (cm)")
    plt.ylabel("Height (m)")
    plt.legend()
    r2_ref = r2_score(df["height_ref"], df["dbh_ref (cm)"])
    r2_est = r2_score(df["Height_m"], df["dbh (cm)"])
    plt.text(
        0.05, 0.95,
        f"$R^2 reference = {r2_ref:.3f}$",
        transform=plt.gca().transAxes,
        verticalalignment='top'
    )
    plt.text(
        0.05, 0.90,
        f"$R^2 estimated = {r2_est:.3f}$",
        transform=plt.gca().transAxes,
        verticalalignment='top'
    )

    plt.savefig(f"{output_folder}/dbh_height_scatter.png", dpi=311)
    
    return

def stem_Qanalysis(df, output_folder):
    """
    Analyses the success rate for stem extraction, and compares it to height classes, per species and per dbh class.
    """
    stem_cov_avg = np.mean(df["Stem_coverage"])
    stem_cov_std = np.std(df["Stem_coverage"])
    plt.figure(figsize=(6, 6))
    plt.scatter(df["Point Count"], df["Stem_coverage"], alpha=0.5)
    r2 = r2_score(df["Point Count"], df["Stem_coverage"])
    plt.text(
        0.05, 0.95,
        f"$R^2 = {r2:.3f}$",
        transform=plt.gca().transAxes,
        verticalalignment='top'
    )
    plt.xlabel("Reference Stem Point Count")
    plt.ylabel("Estimated Stem Coverage")
    plt.savefig(f"{output_folder}/stem_coverage.png", dpi=311)
    stem_cov_r2_tot = r2_score(df["Point Count"], df["Stem_coverage"])
    print(f"Average Stem Coverage: {stem_cov_avg:.2%}, Stem Coverage Std: {stem_cov_std:.2%}, R² for stem coverage compared to point count: {stem_cov_r2_tot:.3f}")
    return stem_cov_avg, stem_cov_std



def biomass_analysis(df, output_folder):
    """
    Compares estimated biomass to reference biomass, computes bias, relative error, and R², and plots relative error vs. reference biomass
    """
    
    df = df.dropna(subset=["Biomass_stem", "Biomass_branch", "Biomass_stem_ref", "Biomass_branch_ref"])
    biomass = df["Biomass_stem"] + df["Biomass_branch"]
    ref_biomass = df["Biomass_stem_ref"] + df["Biomass_branch_ref"]
    bias = np.mean(biomass - ref_biomass)
    bias_std = np.std(biomass - ref_biomass)
    rel_error = np.mean(np.abs(biomass - ref_biomass) / ref_biomass)
    rel_error_std = np.std(np.abs(biomass - ref_biomass) / ref_biomass)
    plt.figure(figsize=(6, 6))
    plt.scatter(ref_biomass, biomass, alpha=0.5)
    r2 = r2_score(ref_biomass, biomass)
    plt.text(
        0.05, 0.95,
        f"$R^2 = {r2:.3f}$",
        transform=plt.gca().transAxes,
        verticalalignment='top'
    )
    plt.ylabel("Estimated AGB (kg)")
    plt.xlabel("Reference AGB (kg)")
    plt.savefig(f"{output_folder}/biomass_error.png", dpi=311)
    print(f"Biomass Bias: {bias:.2f} kg, Biomass Bias Std: {bias_std:.2f} kg, Relative Biomass Error: {rel_error:.2%}, Relative Biomass Error Std: {rel_error_std:.2%}, R²: {r2:.3f}")
    return bias, bias_std, rel_error, rel_error_std, r2



def main():
    ref = gpd.read_file("/mnt/c/Users/digit/Downloads/Examensarbete/Results/matched_trees.gpkg")
    df = pd.read_csv("/mnt/c/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/biomass_height_distribution_summary_0.2.csv")
    ref_subset = ref[["ff3d_tile_id", "ff3d_tile_dist", "ff3d_tile_matched", "DBH_Field", "H_TLS", "Species"]]
    df = df.merge(ref_subset, left_on="TreeID", right_on="ff3d_tile_id", how="left")

    print(df.columns)
    #df = df[df["DBH_Field"] > 10]

    df["dbh_ref (cm)"] = df["DBH_Field"]
    df["height_ref"] = df["H_TLS"]
    df["Species_name"] = df["Species_x"]
    df["dbh (cm)"] = df[["DBH_cm_hlayer_0.1", "DBH_cm_hlayer_0.05", "DBH_cm_hlayer_0.5"]].min(axis=1)
    df = df.drop(columns=["DBH_Field", "H_TLS", "DBH_cm_hlayer_0.1", "DBH_cm_hlayer_0.1", "DBH_cm_hlayer_0.05", "Species_x", "Species_y"])

    #id_to_species = {1: "Pine", 2: "Spruce", 3: "Birch", 7: "Ädel", 11: "Dead" }
   # df["Species_name"] = df["Species"].map(id_to_species)

    bins = [0, 5, 10, 15, 20, 25, 30, 40, 50]
    labels = ['0-5m', '5-10m', '10-15m', '15-20m', '20-25m', '25-30m', '30-40m', '40m+']
    df['height_class'] = pd.cut(df['height_ref'], bins=bins, labels=labels)
    df.loc[df["dbh (cm)"] > 50, "dbh (cm)"] = np.nan

    bins = [0, 5, 10, 15, 20, 25, 30, 40, 50]
    labels = ['0-5cm', '5-10cm', '10-15cm', '15-20cm', '20-25cm', '25-30cm', '30-40cm', '40cm+']
    df['dbh_class'] = pd.cut(df['dbh_ref (cm)'], bins=bins, labels=labels)
    bm_stem = np.zeros(len(df))
    bm_branch = np.zeros(len(df))
    for sp in ["Pine", "Spruce", "Birch"]:
        mask = df["Species_name"] == sp
        stem, branch = MarklundBiomass(
            df.loc[mask, "dbh (cm)"].values * 10,
            sp,
            height_m=df.loc[mask, "Height_m"].values
        )

        if mask.sum() == 0:
            continue

        bm_stem[mask] = stem
        bm_branch[mask] = branch
    df["Biomass_stem"] = bm_stem
    df["Biomass_branch"] = bm_branch


    bm_stem = np.zeros(len(df))
    bm_branch = np.zeros(len(df))
    for sp in ["Pine", "Spruce", "Birch"]:
        mask = df["Species_name"] == sp
        stem, branch = MarklundBiomass(
            df.loc[mask, "dbh_ref (cm)"].values * 10,
            sp,
            height_m=df.loc[mask, "height_ref"].values
        )

        if mask.sum() == 0:
            continue

        bm_stem[mask] = stem
        bm_branch[mask] = branch
    df["Biomass_stem_ref"] = bm_stem
    df["Biomass_branch_ref"] = bm_branch
    print(len(df))
    output_folder = "/mnt/c/Users/digit/Downloads/Examensarbete/Results/biomass_analysis/"
    os.makedirs(output_folder, exist_ok=True)
   
    dbh_success_analysis(df, output_folder)
 
    stem_Qanalysis(df,output_folder)
    
    dbh_analysis(df, output_folder)
    height_analysis(df,output_folder)
    plot_dbh_height(df, output_folder)
    biomass_analysis(df, output_folder)
    print(f"Analysis complete. Results saved to: {output_folder}")

if __name__ == "__main__":
    main()
    
    
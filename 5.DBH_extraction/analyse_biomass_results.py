import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, root_mean_squared_error
import geopandas as gpd
import os
from calculate_biomass import MarklundBiomass
from scipy.stats import linregress
import statsmodels.formula.api as smf

def dbh_extraction_analysis(df, output_folder):
    """
    Analyses the total success rate for dbh extraction, and compares it to height classes, per species and per dbh class.
    """
    df["success"] = df["dbh"].notna() & df["Height_m"].notna()
    dbh_success_rate = len(df[df["success"] == True]) / len(df)
    species_accuracy = df.groupby("Species_name")[f"success"].mean() * 100
    species_accuracy = species_accuracy.sort_values(ascending=False)

    df_ps = df[df["Species_name"].isin(["Pine", "Spruce"])].copy()
    df_ps["success_int"] = df_ps["success"].astype(int)
    df_ps["Species_name"] = pd.Categorical(
        df_ps["Species_name"],
        categories=["Pine", "Spruce"]  # pine becomes reference
    )

    height_accuracy = df.groupby("height_class")[f"success"].mean() * 100
    #height_accuracy = height_accuracy.sort_values(ascending=False)

    dbh_accuracy = df.groupby("dbh_class")[f"success"].mean() * 100
    #dbh_accuracy = dbh_accuracy.sort_values(ascending=False)
    total_dbhs = len(df[df["success"] == True])
    print(f"Overall DBH and Height Extraction Success Rate: {dbh_success_rate:.2%} (Total DBHs: {total_dbhs})")
    print(f"Accuracy by DBH Class:")
    print(dbh_accuracy)
    print(f"Accuracy by Height Class:")
    print(height_accuracy)
    print(f"Species Accuracy:")
    print(species_accuracy)
    return dbh_success_rate, species_accuracy, height_accuracy, dbh_accuracy

def lme(df, output_folder):
    df = df.dropna(subset=["dbh", "dbh_ref", "height_ref", "Species_name"])
    df["dbh_error"] = df["dbh"] - df["dbh_ref"]
    df["abs_dbh_error"] = np.abs(df["dbh_error"])
    bad_trees = df[df["abs_dbh_error"] > 10]["ff3d_tile_id"].unique() 
    print(f"FF3D ids for trees with > 10cm dbh error: {', '.join(map(str, bad_trees))}")
    result = smf.ols(
        "dbh_error ~ Species_name + dbh_ref + height_ref + ff3d_tile_dist ",
        data=df
    ).fit()
    print(result.summary())
    df["residual"] = result.resid
    plt.figure(figsize=(6, 6))
    plt.scatter(df["dbh_ref"], df["residual"], alpha=0.5)
    plt.axhline(0, color="black")
    plt.xlabel("Reference DBH")
    plt.ylabel("Residual error")
    plt.savefig(f"{output_folder}/lms_residuals.png", dpi=311)

def dbh_analysis(df, output_folder):
    """
    Analyses the DBH estimation.
    Compares estimated DBH to reference DBH, computes bias, relative error, and R², and plots relative error vs. reference DBH
    """
    df = df.dropna(subset=["dbh", "dbh_ref"])
    print(f"The amount of DBH for matched trees: {len(df)}")
    dbh = df["dbh"]
    ref_dbh = df["dbh_ref"]
    bias = np.mean(dbh - ref_dbh)
    bias_std = np.std(dbh - ref_dbh)
    rel_error = np.mean(np.abs(dbh - ref_dbh) / ref_dbh)
    rel_error_std = np.std(np.abs(dbh - ref_dbh) / ref_dbh)
    rmse = root_mean_squared_error(ref_dbh, dbh)
    plt.figure(figsize=(6, 6))
    #plt.scatter(ref_dbh, (dbh - ref_dbh) / ref_dbh,  alpha=0.5)
    plt.scatter(ref_dbh, dbh,  alpha=0.5)
    slope, intercept, r_value, p_value, std_err = linregress(ref_dbh, dbh)
    x = np.linspace(min(ref_dbh), max(ref_dbh), 100)
    y = slope * x + intercept
    plt.plot(x, y)
    plt.plot(x, x,  '--', linewidth=1, color='gray', label='y = x')
    r2 = r2_score(ref_dbh, dbh)
    plt.text(
        0.05, 0.95,
        f"$y = {slope:.3f}x + {intercept:.3f}$\n$R^2 = {r2:.3f}$",
        transform=plt.gca().transAxes,
        verticalalignment='top'
    )
    plt.ylabel("Estimated DBH (cm)")
    plt.xlabel("Reference DBH (cm)")
    plt.savefig(f"{output_folder}/dbh.png", dpi=311)
    print(f"DBH Bias: {bias:.2f} cm, DBH Bias Std: {bias_std:.2f} cm, RMSE: {rmse:.2f}, Relative DBH Error: {rel_error:.2%}, Relative DBH Error Std: {rel_error_std:.2%}, R²: {r2:.3f}")
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
    rmse = root_mean_squared_error(ref_height, height)
    plt.figure(figsize=(6, 6))
    plt.scatter(ref_height, height, alpha=0.5)
    slope, intercept, r_value, p_value, std_err = linregress(ref_height, height)
    x = np.linspace(min(ref_height), max(ref_height), 100)
    y = slope * x + intercept
    plt.plot(x, y)
    plt.plot(x,x,  '--', linewidth=1, color='gray', label='y = x')
    plt.ylabel("Estimated Height (m)")
    plt.xlabel("Reference Height (m)")
    r2 = r2_score(ref_height, height)
    plt.text(
        0.05, 0.95,
        f"$y = {slope:.3f}x + {intercept:.3f}$\n$R^2 = {r2:.3f}$",
        transform=plt.gca().transAxes,
        verticalalignment='top'
    )
    plt.savefig(f"{output_folder}/height.png", dpi=311)

    plt.figure(figsize=(6, 6))
    plt.scatter(df["ff3d_tile_dist"], np.abs(height - ref_height) / ref_height, alpha=0.5)
    
    plt.ylabel("Relative Height Error")
    plt.xlabel("FF3D Distance (m)")
    plt.close()
    #plt.savefig(f"{output_folder}/height_error_ff3d.png", dpi=311)

    print(f"Height Bias: {bias:.2f} m, Height Bias Std: {bias_std:.2f} m, RMSE: {rmse:.2f}, Relative Height Error: {rel_error:.2%}, Relative Height Error Std: {rel_error_std:.2%}, R²: {r2:.3f}")
    return bias, bias_std, rel_error, rel_error_std

def plot_dbh_height(df, output_folder):
    """
    Analysing the relationship between DBH and height for both reference and estimated, plotting them against each other, and computing R² for both.
    """
    df = df.dropna(subset=["dbh", "Height_m", "dbh_ref", "height_ref"])
    plt.figure(figsize=(6, 6))
    x_ref = df["dbh_ref"]
    y_ref = df["height_ref"]
    plt.scatter(x_ref, y_ref, label="Reference", alpha=0.5)
    z_ref = np.polyfit(x_ref, y_ref, 1)
    p_ref = np.poly1d(z_ref)
    idx = np.argsort(x_ref)
    plt.plot(
        x_ref.iloc[idx],
        p_ref(x_ref.iloc[idx]),
        label="Reference Trend",
        color="blue"
    )
    r_ref = np.corrcoef(x_ref, y_ref)[0, 1]
    r2_ref = r_ref**2
    x_est = df["dbh"]
    y_est = df["Height_m"]
    plt.scatter(x_est, y_est, label="Estimated", alpha=0.5)
    z_est = np.polyfit(x_est, y_est, 1)
    p_est = np.poly1d(z_est)
    idx = np.argsort(x_est)
    plt.plot(
        x_est.iloc[idx],
        p_est(x_est.iloc[idx]),
        label="Estimated Trend",
        color="orange"
    )
    r_est = np.corrcoef(x_est, y_est)[0, 1]
    r2_est = r_est**2
    plt.xlabel("DBH ")
    plt.ylabel("Height (m)")

    plt.text(
        0.05, 0.95,
        f"$R^2_{{reference}} = {r2_ref:.3f}$",
        transform=plt.gca().transAxes,
        verticalalignment='top',
        color='blue'
    )
    plt.text(
        0.05, 0.90,
        f"$R^2_{{estimated}} = {r2_est:.3f}$",
        transform=plt.gca().transAxes,
        verticalalignment='top',
        color='orange'
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_folder}/dbh_height_scatter.png", dpi=311)
    plt.close()

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
    print(f"Average Stem Coverage: {stem_cov_avg:.2%}, Stem Coverage Std: {stem_cov_std:.2%}")
    return stem_cov_avg, stem_cov_std

def plot_dbh_error_vs_height_error(df, output_folder):
    """
    Plots DBH estimation error against height estimation error.
    """

    df = df.dropna(subset=["dbh", "dbh_ref", "Height_m", "height_ref"]).copy()

    # Absolute errors
    df["dbh_error"] = df["dbh"] - df["dbh_ref"]
    df["height_error"] = df["Height_m"] - df["height_ref"]

    # Relative errors
    df["dbh_rel_error"] = (df["dbh"] - df["dbh_ref"]) / df["dbh_ref"]
    df["height_rel_error"] = (df["Height_m"] - df["height_ref"]) / df["height_ref"]

    # ---------------------------------
    # Absolute error plot
    # ---------------------------------
    plt.figure(figsize=(6, 6))

    species_colors = {
        "Pine": "orange",
        "Spruce": "green",
        "Birch": "blue"
    }

    for sp, group in df.groupby("Species_name"):
        plt.scatter(
            group["height_error"],
            group["dbh_error"],
            alpha=0.5,
            label=sp,
            color=species_colors.get(sp, "gray")
        )

    plt.legend()

    slope, intercept, r_value, p_value, std_err = linregress(
        df["height_error"],
        df["dbh_error"]
    )

    x = np.linspace(
        df["height_error"].min(),
        df["height_error"].max(),
        100
    )

    y = slope * x + intercept

    plt.plot(x, y)

    r2 = r_value**2

    plt.axhline(0, linestyle="--", color="gray", linewidth=1)
    plt.axvline(0, linestyle="--", color="gray", linewidth=1)

    plt.xlabel("Height Error (m)")
    plt.ylabel("DBH Error (cm)")
    

    plt.text(
        0.05, 0.95,
        f"$R^2 = {r2:.3f}$\n$p = {p_value:.3e}$",
        transform=plt.gca().transAxes,
        verticalalignment='top'
    )

    plt.tight_layout()
    plt.savefig(f"{output_folder}/dbh_error_vs_height_error.png", dpi=311)
    plt.close()

    # ---------------------------------
    # Relative error plot
    # ---------------------------------
    plt.figure(figsize=(6, 6))

    plt.scatter(
        df["height_rel_error"],
        df["dbh_rel_error"],
        alpha=0.5
    )

    slope, intercept, r_value, p_value, std_err = linregress(
        df["height_rel_error"],
        df["dbh_rel_error"]
    )

    x = np.linspace(
        df["height_rel_error"].min(),
        df["height_rel_error"].max(),
        100
    )

    y = slope * x + intercept

    plt.plot(x, y)

    r2 = r_value**2

    plt.axhline(0, linestyle="--", color="gray", linewidth=1)
    plt.axvline(0, linestyle="--", color="gray", linewidth=1)

    plt.xlabel("Relative Height Error")
    plt.ylabel("Relative DBH Error")

    plt.text(
        0.05, 0.95,
        f"$R^2 = {r2:.3f}$\n$p = {p_value:.3e}$",
        transform=plt.gca().transAxes,
        verticalalignment='top'
    )

    plt.tight_layout()
    plt.savefig(f"{output_folder}/dbh_rel_error_vs_height_rel_error.png", dpi=311)
    plt.close()

    print(
        f"DBH vs Height Error Correlation:\n"
        f"Absolute error R²: {r2:.3f}, p={p_value:.3e}"
    )

def biomass_analysis(df, output_folder):
    """
    Compares estimated biomass to reference biomass, computes bias, relative error, and R², and plots relative error vs. reference biomass
    """
    df.loc[df["Biomass_stem"] == 0, "Biomass_stem"] = np.nan
    df = df.dropna(subset=["Biomass_stem", "Biomass_branch", "Biomass_stem_ref", "Biomass_branch_ref"])
    biomass = df["Biomass_stem"] + df["Biomass_branch"]
    ref_biomass = df["Biomass_stem_ref"] + df["Biomass_branch_ref"]
    bias = np.mean(biomass - ref_biomass)
    bias_std = np.std(biomass - ref_biomass)
    rel_error = np.mean(np.abs(biomass - ref_biomass) / ref_biomass)
    rel_error_std = np.std(np.abs(biomass - ref_biomass) / ref_biomass)
    rmse = root_mean_squared_error(ref_biomass, biomass)
    plt.figure(figsize=(6, 6))
    plt.scatter(ref_biomass, biomass, alpha=0.5)
    slope, intercept, r_value, p_value, std_err = linregress(ref_biomass, biomass)
    x = np.linspace(min(ref_biomass), max(ref_biomass), 100)
    y = slope * x + intercept
    plt.plot(x, y)
    plt.plot(x,x,  '--', linewidth=1, color='gray', label='y = x')
    r2 = r2_score(ref_biomass, biomass)
    plt.text(
        0.05, 0.95,
        f"$y = {slope:.3f}x + {intercept:.3f}$\n$R^2 = {r2:.3f}$",
        transform=plt.gca().transAxes,
        verticalalignment='top'
    )
    plt.ylabel("Estimated AGB (kg)")
    plt.xlabel("Reference AGB (kg)")
    plt.savefig(f"{output_folder}/biomass_error.png", dpi=311)

    print(f"Biomass Bias: {bias:.2f} kg, Biomass Bias Std: {bias_std:.2f} kg, RMSE: {rmse:.2f}, Relative Biomass Error: {rel_error:.2%}, Relative Biomass Error Std: {rel_error_std:.2%}, R²: {r2:.3f}")
    return bias, bias_std, rel_error, rel_error_std, r2



def main():
    ref = gpd.read_file("/mnt/c/Users/digit/Downloads/Examensarbete/Examensarbete/5.DBH_extraction/matched_trees.gpkg")
    df = pd.read_csv("/mnt/c/Users/digit/Downloads/Examensarbete/Examensarbete/5.DBH_extraction/biomass_height_distribution_summary_0.2.csv")
    ref_subset = ref[["ff3d_tile_id", "ff3d_tile_dist", "ff3d_tile_matched", "DBH_Field", "H_TLS", "Species"]]
    df = df.merge(ref_subset, left_on="TreeID", right_on="ff3d_tile_id", how="left")
    #df = df[df["DBH_Field"] > 10]

    df["dbh_ref"] = df["DBH_Field"]
    df["height_ref"] = df["H_TLS"]
    df["Species_name"] = df["Species_x"]
    df["dbh"] = df[["DBH_cm_hlayer_0.1", "DBH_cm_hlayer_0.05", "DBH_cm_hlayer_0.5"]].min(axis=1)
    df = df.drop(columns=["DBH_Field", "H_TLS", "DBH_cm_hlayer_0.1", "DBH_cm_hlayer_0.5", "DBH_cm_hlayer_0.05", "Species_x", "Species_y"])
    df["dbh_error"] = df["dbh"] - df["dbh_ref"]
    df["abs_dbh_error"] = np.abs(df["dbh_error"])
    bad_trees = df[df["abs_dbh_error"] > 10]["ff3d_tile_id"].unique() 
    #df = df[~df["ff3d_tile_id"].isin(bad_trees)]
    
    #id_to_species = {1: "Pine", 2: "Spruce", 3: "Birch", 7: "Ädel", 11: "Dead" }
    #df["Species_name"] = df["Species"].map(id_to_species)

    bins = [0, 5, 10, 15, 20, 25, 30, 40, 50]
    labels = ['0-5m', '5-10m', '10-15m', '15-20m', '20-25m', '25-30m', '30-40m', '40m+']
    df['height_class'] = pd.cut(df['height_ref'], bins=bins, labels=labels)
    df.loc[df["dbh"] > 50, "dbh"] = np.nan
    df.loc[df["Species_name"] == "Unknown", "Species_name"] = "Other"

    bins = [0, 5, 10, 15, 20, 25, 30, 40, 50]
    labels = ['0-5cm', '5-10cm', '10-15cm', '15-20cm', '20-25cm', '25-30cm', '30-40cm', '40cm+']
    df['dbh_class'] = pd.cut(df['dbh_ref'], bins=bins, labels=labels)

    bm_stem = np.zeros(len(df))
    bm_branch = np.zeros(len(df))
    for sp in ["Pine", "Spruce", "Birch"]:
        mask = df["Species_name"] == sp
        stem, branch = MarklundBiomass(
            df.loc[mask, "dbh_ref"].values * 10,
            sp,
            height_m=df.loc[mask, "height_ref"].values
        )

        if mask.sum() == 0:
            continue

        bm_stem[mask] = stem
        bm_branch[mask] = branch
    df["Biomass_stem_ref"] = bm_stem
    df["Biomass_branch_ref"] = bm_branch
    output_folder = "/mnt/c/Users/digit/Downloads/tets/"
    os.makedirs(output_folder, exist_ok=True)
   
    dbh_extraction_analysis(df, output_folder)
    #stem_Qanalysis(df,output_folder)
    dbh_analysis(df, output_folder)
    height_analysis(df,output_folder)
    plot_dbh_height(df, output_folder)
    biomass_analysis(df, output_folder)
    plot_dbh_error_vs_height_error(df, output_folder)
    lme(df, output_folder)
    print(f"Analysis complete. Results saved to: {output_folder}")

if __name__ == "__main__":
    main()
    
    
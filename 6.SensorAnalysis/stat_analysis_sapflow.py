import os
import numpy as np
import pandas as pd

from scipy.stats import (
    ttest_rel,
    wilcoxon,
    shapiro,
    friedmanchisquare
)

import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM

# ==========================================================
# SETTINGS
# ==========================================================

VOXEL_SIZES = [0.1, 0.2, 0.4]

VOXEL_PARQUET = (
    "/mnt/c/Users/digit/Downloads/Examensarbete/Results/"
    "voxel_biomass.parquet"
)

SAPFLOW_CSV = (
    "/mnt/c/Users/digit/Downloads/Examensarbete/Data/"
    "sapflow_measurements.csv"
)

OUTPUT_DIR = (
    "/mnt/c/Users/digit/Downloads/Examensarbete/Results/"
    "sapflow_analysis/"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

voxels = pd.read_parquet(VOXEL_PARQUET)

sap = pd.read_csv(SAPFLOW_CSV)

# ----------------------------------------------------------
# EXPECTED SAPFLOW FORMAT
# ----------------------------------------------------------
#
# tree_id,height,direction,sap_flow
#
# Example:
#
# 12,1.3,N,23.1
# 12,1.3,S,18.7
# 12,8,N,12.4
# 12,8,S,10.1
#
# ----------------------------------------------------------

sap["direction"] = sap["direction"].str.upper()

# ==========================================================
# KEEP FOLIAGE ONLY
# ==========================================================

foliage = voxels[
    (voxels["component"] == "foliage") &
    (voxels["direction"].notna())
].copy()

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def paired_test(df, col1, col2, label=""):

    paired = df[[col1, col2]].dropna()

    if len(paired) < 3:
        print(f"{label}: too few samples")
        return

    diff = paired[col1] - paired[col2]

    shapiro_p = shapiro(diff).pvalue

    print(f"\n{label}")
    print(f"N = {len(paired)}")
    print(f"Normality p = {shapiro_p:.4f}")

    if shapiro_p > 0.05:

        stat, p = ttest_rel(
            paired[col1],
            paired[col2]
        )

        print("Paired t-test")
        print(f"t = {stat:.4f}")
        print(f"p = {p:.6f}")

    else:

        stat, p = wilcoxon(
            paired[col1],
            paired[col2]
        )

        print("Wilcoxon signed-rank")
        print(f"W = {stat:.4f}")
        print(f"p = {p:.6f}")


# ==========================================================
# MAIN ANALYSIS
# ==========================================================

all_model_results = []

for voxel_size in VOXEL_SIZES:

    print_section(f"VOXEL SIZE {voxel_size}")

    vox = foliage[
        foliage["voxel_size"] == voxel_size
    ].copy()

    # ======================================================
    # 1. TOTAL NORTH/SOUTH BIOMASS
    # ======================================================

    biomass_ns = (
        vox.groupby(
            ["tree_id", "direction"]
        )["biomass"]
        .sum()
        .unstack()
        .reset_index()
    )

    print_section("N/S foliage biomass comparison")

    paired_test(
        biomass_ns,
        "N",
        "S",
        label=f"Biomass N vs S ({voxel_size} m)"
    )

    # ======================================================
    # 2. SAP FLOW N/S COMPARISON
    # ======================================================

    sap_ns = (
        sap.groupby(
            ["tree_id", "direction"]
        )["sap_flow"]
        .mean()
        .unstack()
        .reset_index()
    )

    print_section("N/S sap flow comparison")

    paired_test(
        sap_ns,
        "N",
        "S",
        label=f"Sap flow N vs S ({voxel_size} m)"
    )

    # ======================================================
    # 3. SAP FLOW DIFFERENCE BETWEEN HEIGHTS
    # ======================================================

    sap_height = (
        sap.groupby(
            ["tree_id", "height"]
        )["sap_flow"]
        .mean()
        .reset_index()
    )

    pivot_height = sap_height.pivot(
        index="tree_id",
        columns="height",
        values="sap_flow"
    )

    print_section("Height effect on sap flow")

    valid = pivot_height.dropna()

    if len(valid) > 3:

        stat, p = friedmanchisquare(
            valid[1.3],
            valid[8],
            valid[14]
        )

        print("Friedman repeated-measures test")
        print(f"Statistic = {stat:.4f}")
        print(f"p = {p:.6f}")

    # ======================================================
    # 4. BIOMASS ABOVE SENSOR HEIGHT
    # ======================================================

    sensor_heights = [1.3, 8, 14]

    biomass_above = []

    for h in sensor_heights:

        temp = (
            vox[vox["z"] > h]
            .groupby(
                ["tree_id", "direction"]
            )["biomass"]
            .sum()
            .reset_index()
        )

        temp["height"] = h

        biomass_above.append(temp)

    biomass_above = pd.concat(
        biomass_above,
        ignore_index=True
    )

    # ======================================================
    # 5. MERGE WITH SAP FLOW
    # ======================================================

    model_df = sap.merge(
        biomass_above,
        on=["tree_id", "height", "direction"],
        how="left"
    )

    model_df.rename(
        columns={
            "biomass": "foliage_above"
        },
        inplace=True
    )

    model_df["direction_bin"] = (
        model_df["direction"] == "N"
    ).astype(int)

    # ======================================================
    # 6. MIXED EFFECT MODEL
    # ======================================================

    print_section("Mixed effects model")

    try:

        model = smf.mixedlm(
            (
                "sap_flow ~ "
                "height + "
                "direction_bin + "
                "foliage_above"
            ),
            model_df,
            groups=model_df["tree_id"]
        )

        result = model.fit()

        print(result.summary())

        out_path = (
            f"{OUTPUT_DIR}/"
            f"mixed_model_{voxel_size}.txt"
        )

        with open(out_path, "w") as f:
            f.write(result.summary().as_text())

        all_model_results.append(result)

    except Exception as e:

        print("Model failed:")
        print(e)

    # ======================================================
    # 7. INTERACTION MODEL
    # ======================================================

    print_section("Interaction model")

    try:

        model2 = smf.mixedlm(
            (
                "sap_flow ~ "
                "height * direction_bin + "
                "foliage_above"
            ),
            model_df,
            groups=model_df["tree_id"]
        )

        result2 = model2.fit()

        print(result2.summary())

        out_path = (
            f"{OUTPUT_DIR}/"
            f"interaction_model_{voxel_size}.txt"
        )

        with open(out_path, "w") as f:
            f.write(result2.summary().as_text())

    except Exception as e:

        print("Interaction model failed:")
        print(e)

    # ======================================================
    # 8. TREE-SPECIFIC ASYMMETRY
    # ======================================================

    print_section("Asymmetry analysis")

    asym = biomass_ns.copy()

    asym["biomass_ratio"] = (
        asym["N"] / asym["S"]
    )

    sap_asym = sap_ns.copy()

    sap_asym["sap_ratio"] = (
        sap_asym["N"] / sap_asym["S"]
    )

    asym_df = asym.merge(
        sap_asym,
        on="tree_id",
        suffixes=("_bio", "_sap")
    )

    try:

        corr = asym_df[
            ["biomass_ratio", "sap_ratio"]
        ].corr()

        print(corr)

        corr.to_csv(
            f"{OUTPUT_DIR}/"
            f"asymmetry_correlation_{voxel_size}.csv"
        )

    except Exception as e:

        print(e)

# ==========================================================
# OPTIONAL EXTRA ANALYSES
# ==========================================================

print_section("SUGGESTED EXTRA ANALYSES")

print("""
1. Species-specific models
   sap_flow ~ height + direction + foliage_above + species

2. Nonlinear height response
   Use splines or GAMs

3. Relative biomass instead of absolute
   foliage_above / total_foliage

4. Vertical foliage distribution metrics:
   - center of mass
   - skewness
   - canopy depth

5. Lagged sap flow relationships
   if temporal measurements exist

6. Random slopes mixed models:
   allow height response to vary by tree

7. Compare voxel resolutions:
   AIC/BIC between voxel sizes

8. Spatial autocorrelation:
   Does local canopy clustering affect sap flow?

9. Include stem biomass above sensor

10. Wind/light exposure asymmetry:
   compare with crown directionality
""")
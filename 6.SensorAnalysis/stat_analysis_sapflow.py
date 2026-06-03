import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

import statsmodels.formula.api as smf

# ==========================================================
# SETTINGS
# ==========================================================

VOXEL_SIZES = [0.1, 0.2, 0.4]

VOXEL_PARQUET = (
    "/mnt/c/Users/digit/Downloads/Examensarbete/Results/saptrees_segment/"
    "voxel_biomass_saptrees.parquet"
)

SAPFLOW_CSV = (
    "/mnt/c/Users/digit/Downloads/Examensarbete/Data/Sap_flow_data"
    "/Stack of Subset of radar_47355_one_res_Feb22_2026.txt"
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
colors = {
    "stem": "saddlebrown",
    "foliage": "forestgreen"
}

for tree_id in sorted(voxels["tree_id"].unique()):

    tree_data = voxels[voxels["tree_id"] == tree_id]
    voxel_sizes = sorted(tree_data["voxel_size"].unique())

    fig = plt.figure(figsize=(7 * len(voxel_sizes), 7))

    for i, voxel_size in enumerate(voxel_sizes, start=1):

        ax = fig.add_subplot(
            1, len(voxel_sizes), i,
            projection="3d"
        )

        subset = tree_data[tree_data["voxel_size"] == voxel_size]

        # Draw each voxel cube
        for _, row in subset.iterrows():

            s = row["voxel_size"]

            # Convert center coordinates to cube corner coordinates
            x0 = row["x"] - s / 2
            y0 = row["y"] - s / 2
            z0 = row["z"] - s / 2

            ax.bar3d(
                x0, y0, z0,
                s, s, s,
                color=colors.get(row["component"], "gray"),
                alpha=0.9,
                shade=True,
                linewidth=0
            )

        # Equal aspect ratio
        xmin, xmax = subset["x"].min(), subset["x"].max()
        ymin, ymax = subset["y"].min(), subset["y"].max()
        zmin, zmax = subset["z"].min(), subset["z"].max()

        max_range = max(
            xmax - xmin,
            ymax - ymin,
            zmax - zmin
        )

        xmid = (xmin + xmax) / 2
        ymid = (ymin + ymax) / 2
        zmid = (zmin + zmax) / 2

        ax.set_xlim(xmid - max_range / 2, xmid + max_range / 2)
        ax.set_ylim(ymid - max_range / 2, ymid + max_range / 2)
        ax.set_zlim(zmid - max_range / 2, zmid + max_range / 2)

        ax.set_title(f"Voxel size = {voxel_size} m")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Better viewing angle
        ax.view_init(elev=25, azim=45)

    # Shared legend
    legend_elements = [
        Patch(facecolor=colors["stem"], label="Stem"),
        Patch(facecolor=colors["foliage"], label="Foliage")
    ]
    fig.legend(handles=legend_elements, loc="upper right")

    #fig.suptitle(f"Tree {tree_id}", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/voxel_data_{tree_id}.png", dpi=300, bbox_inches="tight")
    plt.close()

sap = pd.read_csv(SAPFLOW_CSV, parse_dates=["TIMESTAMP"])

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
def direction_short(d):
    if d == "North":
        return "N"
    elif d == "South":
        return "S"
    return None


def treeid(t):
    if t.split("_")[0] == "Pine":
        return "tree403"
    elif t.split("_")[0] == "Spruce":
        return "tree408"
    return None


sap["direction"] = sap["Orientation"].apply(direction_short)
sap["height"] = sap["Height"].str.split(" ").str[0].astype(float)
sap["tree_id"] = sap["Label"].apply(treeid)
sap["sap_flow"] = sap["Mean Js"].astype(float)
sap.drop(columns=["Orientation", "Height", "Mean Js"], inplace=True)

print(sap.head())
print(voxels.head())

sap["date"] = sap["TIMESTAMP"].dt.date

# --- Calculate daily mean sap flow per sensor ---
daily_mean = (
    sap.groupby(["Label", "date"], as_index=False)["sap_flow"]
    .mean()
)
daily_mean = (
    sap.groupby(
        ["tree_id", "Label", "direction", "height", "date"],
        as_index=False
    )["sap_flow"]
    .mean()
)
north_colors = {
    1: "#84c1e1",   # light blue
    8: "#3182bd",   # medium blue
    15: "#08519c",  # dark blue
}

south_colors = {
    1: "#f1de4f",   # light yellow
    8: "#fec44f",   # medium yellow
    15: "#d95f0e",  # dark orange/yellow
}

# --- One plot per tree ---
tree_ids = daily_mean["tree_id"].unique()

for tree_id in tree_ids:

    tree_data = daily_mean[
        daily_mean["tree_id"] == tree_id
    ]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot all sensors for this tree
    for label in tree_data["Label"].unique():

        sensor_data = (
            tree_data[tree_data["Label"] == label]
            .sort_values("date")
        )

        direction = sensor_data["direction"].iloc[0]
        height = sensor_data["height"].iloc[0]
        if direction == "N":
            color = north_colors[int(height)]
        else:
            color = south_colors[int(height)]
        ax.plot(
            sensor_data["date"],
            sensor_data["sap_flow"],
            label=f"{direction} {height} m",
            color=color
        )

    ax.set_title(f"Daily Mean Sap Flow — {tree_id}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean daily sap flow")
    ax.legend(title="Sensor")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/daily_sap_flow_by_sensor_{tree_id}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
# ==========================================================
# KEEP FOLIAGE ONLY
# ==========================================================

foliage = voxels[
    (voxels["component"] == "foliage") &
    (voxels["direction"].notna())
].copy()
# ==========================================================
# ONE COMBINED VERTICAL DISTRIBUTION PLOT
# ==========================================================

tree_ids = ["tree403", "tree408"]

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 9),
    sharey=True
)

for ax, tree_id in zip(axes, tree_ids):

    for voxel_size in VOXEL_SIZES:

        tree_data = (
            foliage[
                (foliage["tree_id"] == tree_id) &
                (foliage["voxel_size"] == voxel_size)
            ]
            .groupby("z")["biomass"]
            .sum()
            .reset_index()
            .sort_values("z")
        )
        tree_data["biomass_density"] = (
            tree_data["biomass"] / voxel_size
        )

        ax.plot(
            tree_data["biomass_density"],
            tree_data["z"],
            linewidth=2,
            label=f"{voxel_size} m"
        )

    ax.set_title(tree_id, fontsize=14)
    ax.set_xlabel("Foliage biomass density (kg/m³)")
    ax.grid(True)
    ax.legend(title="Voxel size")

axes[0].set_ylabel("Height (m)")
fig.suptitle(
    "Vertical foliage biomass distribution",
    fontsize=16
)
plt.tight_layout()
plot_path = (
    f"{OUTPUT_DIR}/"
    "vertical_foliage_distribution_combined.png"
)
plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)
plt.close()
print(f"Saved combined plot: {plot_path}")

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def paired_test(df, col1, col2, label=""):

    paired = df[["tree_id", col1, col2]].dropna()
    print(paired.head())
    diff = paired[col2] / paired[col1]
    print(f"Differences between {col1} and {col2}:")
    print(diff)



# ==========================================================
# MAIN ANALYSIS
# ==========================================================


sap = sap[sap["TIMESTAMP"] < "2025-09-01"]
for voxel_size in VOXEL_SIZES:
    all_model_results = {}
    print_section(f"VOXEL SIZE {voxel_size}")

    vox = foliage[
        foliage["voxel_size"] == voxel_size
    ].copy()

    # ======================================================
    # 1. TOTAL NORTH/SOUTH BIOMASS check
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
    # 2. SAP FLOW N/S COMPARISON check
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
    print(valid.head())

    # ======================================================
    # 4. BIOMASS ABOVE SENSOR HEIGHT
    # ======================================================

    sensor_heights = [1.3, 8, 15]

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
    print(biomass_above.head())

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
    model_df = (
        model_df.groupby(
            [
                "tree_id",
                "height",
                "direction",
                "foliage_above"
            ]
        )["sap_flow"]
        .mean()
        .reset_index()
    )

    model_df["direction_bin"] = (
            model_df["direction"] == "N"
        ).astype(int)
    print_section(
        "Plotting sap flow vs foliage above sensor (tree comparison)"
    )

    plot_df = (
        model_df.groupby(
            [
                "tree_id",
                "height",
                "direction",
                "foliage_above"
            ]
        )["sap_flow"]
        .mean()
        .reset_index()
    )

    trees = plot_df["tree_id"].unique()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        sharey=True,
        sharex=True
    )

    for ax, tree in zip(axes, trees):

        sub = plot_df[plot_df["tree_id"] == tree]

        sns.scatterplot(
            data=sub,
            x="foliage_above",
            y="sap_flow",
            hue="direction",
            style="height",
            s=80,
            ax=ax
        )

        sns.lineplot(
            data=sub.sort_values("foliage_above"),
            x="foliage_above",
            y="sap_flow",
            hue="direction",
            style="height",
            legend=False,
            ax=ax
        )

        ax.set_title(f"Tree {tree}")
        ax.set_xlabel("Foliage biomass above sensor")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Mean sap flow")

    # move legend outside (only once)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend_.remove()
    axes[1].legend(handles, labels, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.suptitle(
        f"Sap flow vs foliage above sensor (voxel size = {voxel_size} m)",
        y=1.02
    )

    plt.tight_layout()

    plot_path = (
        f"{OUTPUT_DIR}/"
        f"sapflow_vs_foliage_bytree_{voxel_size}.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
    # ======================================================
    # 6. MIXED EFFECT MODEL
    # ======================================================

    print_section("Mixed effects model")

    try:
        for param in ["height", "direction_bin", "foliage_above"]:
            for tree_id in model_df["tree_id"].unique():
                df_sample = model_df[model_df["tree_id"] == tree_id]
                model = smf.ols(f"sap_flow ~ {param}", data=df_sample)

                result = model.fit()
                print(result.summary())

                out_path = (
                    f"{OUTPUT_DIR}/"
                    f"mixed_model_{voxel_size}_{param}_{tree_id}.txt"
                )

                with open(out_path, "w") as f:
                    f.write(f"Model for {param} (tree {tree_id}):\n\n")
                    f.write(result.summary().as_text())

                all_model_results[(param, tree_id)] = result

    except Exception as e:

        print("Model failed:")
        print(e)


    out_path = (
                f"{OUTPUT_DIR}/"
                f"all_models_{voxel_size}.txt"
            )

    with open(out_path, "w") as f:
        for r in all_model_results:
            f.write(f"Model for {r[0]} (tree {r[1]}):\n\n")
            f.write(all_model_results[r].summary().as_text())

import geopandas as gpd
import numpy as np

def MarklundBiomass(diameter_mm, species, height_m=None):
    """
    Calculate above-ground tree biomass (stem + branch) using Marklund models.

    Parameters
    ----------
    diameter_mm : float or array-like
        Diameter at breast height in mm
    species : str
        "Pinus", "Picea", or "Deciduous"
    height_m : float or array-like, optional
        Tree height in meters. If None, DBH-only models are used.

    Returns
    -------
    biomass_total : float or array-like
        Total above-ground biomass (stem + branch)
    """

    d = diameter_mm #/ 10.0  # convert mm → cm

    if species == "Pine":

        if height_m is None:
            # Pine: DBH only
            bm_stem = np.exp(11.3264 * d / (d + 13.0) - 2.3388)        # T1
            bm_branch = np.exp(9.1015 * d / (d + 10.0) - 2.8604)      # T13
        else:
            # Pine: DBH + height
            bm_stem = np.exp(
                7.5939 * d / (d + 13.0)
                + 0.0151 * height_m
                + 0.8799 * np.log(height_m)
                - 2.6768
            )  # T2

            bm_branch = np.exp(
                13.3955 * d / (d + 10.0)
                - 1.1955 * np.log(height_m)
                - 2.5413
            )  # T14

        return bm_stem, bm_branch

    elif species == "Spruce":

        if height_m is None:
            # Spruce: DBH only
            bm_stem = np.exp(11.3341 * d / (d + 14.0) - 2.0571)        # G1
            bm_branch = np.exp(8.5242 * d / (d + 13.0) - 1.2804)      # G11
        else:
            # Spruce: DBH + height
            bm_stem = np.exp(
                7.4690 * d / (d + 14.0)
                + 0.0289 * height_m
                + 0.6828 * np.log(height_m)
                - 2.1702
            )  # G2

            bm_branch = np.exp(
                10.9708 * d / (d + 13.0)
                - 0.0124 * height_m
                - 0.4923 * np.log(height_m)
                - 1.2063
            )  # G12

        return bm_stem, bm_branch

    elif species == "Birch":

        if height_m is None:
            bm_stem = np.exp(11.0735 * d / (d + 8.0) - 3.0932)         # B1
            bm_branch = np.exp(10.2806 * d / (d + 10.0) - 3.3633)     # B2
        else:
            bm_stem = np.exp(
                8.2827 * d / (d + 7.0)
                + 0.0393 * height_m
                + 0.5772 * np.log(height_m)
                - 3.5686
            )

            bm_branch = np.exp(
                10.2806 * d / (d + 10.0)
                - 3.3633
            )

        return bm_stem, bm_branch

    else:
        raise ValueError("Species must be 'Pinus', 'Picea', or 'Deciduous'")

def calculate_biomass(tree_df):
    # Prepare output arrays
    bm_stem = np.zeros(len(tree_df))
    bm_branch = np.zeros(len(tree_df))

    for sp in ["Pine", "Spruce", "Birch"]:
        mask = tree_df["Species"] == sp

        if mask.sum() == 0:
            continue

        stem, branch = MarklundBiomass(
            tree_df.loc[mask, "DBH_Field"].values,  # <-- NO /10 here
            sp
        )

        bm_stem[mask] = stem
        bm_branch[mask] = branch

    return bm_stem, bm_branch


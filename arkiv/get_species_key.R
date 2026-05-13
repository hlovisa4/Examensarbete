library(lidR)
library(sf)
library(dplyr)

# -----------------------------
# Read Files
# -----------------------------
las_file_ff3d_full <- "C:/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/tile_ALS_clipped_to_reflist.las" 
ref_file <- "C:/Users/digit/Downloads/Examensarbete/Data/TreesTowerFoot240829.gpkg" 
out_file <- "C:/Users/digit/Downloads/Examensarbete/Results/ff3d_match_key.csv"

max_dist <- 6   # maximum allowed distance (m)


mycsf <- csf(sloop_smooth = FALSE, class_threshold = 0.5, cloth_resolution = 0.5, time_step = 0.65)

ref <- st_read(ref_file)
ref <- ref[!duplicated(ref$GlobalID), ]

process_las_key <- function(las_path, ref, max_dist) {
  
  las <- readLAS(las_path)
  las <- classify_ground(las, mycsf)
  las <- normalize_height(las, knnidw())
  
  df <- as.data.frame(las@data)
  
  # -----------------------------
  # ff3d tree centroids
  # -----------------------------
  ff3d <- df %>%
    group_by(instance_pred) %>%
    summarise(
      x = mean(X),
      y = mean(Y),
      height = max(Z),
      .groups = "drop"
    )
  
  ff3d_sf <- st_as_sf(ff3d, coords = c("x", "y"), crs = st_crs(ref))
  
  # -----------------------------
  # nearest ref tree
  # -----------------------------
  nn_idx <- st_nearest_feature(ff3d_sf, ref)
  nn_dist <- as.numeric(
    st_distance(ff3d_sf, ref[nn_idx, ], by_element = TRUE)
  )

  
  # -----------------------------
  # build key table
  # -----------------------------
  out <- data.frame(
    ff3d_id = ff3d$instance_pred,
    ref_id = ref$GlobalID[nn_idx],
    species = ref$Species[nn_idx],
    distance = nn_dist
  )
  

  
  return(out)
}

ff3d_key <- process_las_key(
  las_file_ff3d_full,
  ref
)
write.csv(ff3d_key, out_file, row.names = FALSE)

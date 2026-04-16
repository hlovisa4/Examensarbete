library(lidR)
library(sf)
library(dplyr)

# -----------------------------
# Read Files
# -----------------------------
las_file_ff3d <- "C:/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/tls_als_ff3d_2.las"
las_file_lidr <-  "C:/Users/digit/Downloads/Examensarbete/Results/tls_als_lidr_2.las"
ref_file <- "C:/Users/digit/Downloads/Examensarbete/Data/TreesTowerFoot240829.gpkg"
out_file <- "C:/Users/digit/Downloads/Examensarbete/Results/matched_trees_als_tls.gpkg"

max_dist <- 2   # maximum allowed distance (m)


lasFF3D <- readLAS(las_file_ff3d)
lasCHM <- readLAS(las_file_lidr)
mycsf <- csf(sloop_smooth = FALSE, class_threshold = 0.5, cloth_resolution = 0.5, time_step = 0.65)
lasFF3D_csf <- classify_ground(lasFF3D, mycsf)
lasFF3D <- normalize_height(lasFF3D_csf, knnidw())
lasCHM_csf <- classify_ground(lasCHM, mycsf)
lasCHM <- normalize_height(lasCHM_csf, knnidw())

ref <- st_read(ref_file)
ref <- ref[!duplicated(ref$GlobalID), ]

process_las <- function(las, ref, max_dist, prefix) {
  
  df <- as.data.frame(las@data)
  
  # -----------------------------
  # Centroids
  # -----------------------------
  centroids <- df %>%
    group_by(instance_pred) %>%
    summarise(
      x = mean(X),
      y = mean(Y),
      .groups = "drop"
    )
  
  # -----------------------------
  # Heights
  # -----------------------------
  heights <- df %>%
    group_by(instance_pred) %>%
    summarise(height = max(Z), .groups = "drop")
  
  trees <- left_join(centroids, heights, by = "instance_pred")
  trees_sf <- st_as_sf(trees, coords = c("x","y"), crs = st_crs(ref))
  
  # -----------------------------
  # Matching
  # -----------------------------
  nn_idx <- st_nearest_feature(ref, trees_sf)
  nn_dist <- as.numeric(st_distance(ref, trees_sf[nn_idx,], by_element = TRUE))
  
  matched <- nn_dist <= max_dist
  
  # -----------------------------
  # Return as dataframe
  # -----------------------------
  out <- data.frame(
    matched = as.integer(matched),
    dist = nn_dist,
    id = NA,
    height = NA,
    height_ratio = NA
  )
  
  out$id[matched] <- trees$instance_pred[nn_idx[matched]]
  out$height[matched] <- trees$height[nn_idx[matched]]
  out$height_ratio[matched] <- trees$height[nn_idx[matched]] / ref$H_TLS[matched]
  
  # add prefix to column names
  colnames(out) <- paste0(prefix, "_", colnames(out))
  
  return(out)
}

ff3d_res <- process_las(lasFF3D, ref, max_dist, "ff3d")
chm_res  <- process_las(lasCHM, ref, max_dist, "chm")
ref <- bind_cols(ref, ff3d_res, chm_res)
st_write(ref, out_file, delete_dsn = TRUE)


library(lidR)
library(sf)
library(dplyr)

# -----------------------------
# Read Files
# -----------------------------
las_file_ff3d <- "C:/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/remerged_ff3d_segmented_cloud_plus_missing_points.las"
las_file_lidr <-  "C:/Users/digit/Downloads/Examensarbete/Results/ALS_lidr_segmentation.las"
ref_file <- "C:/Users/digit/Downloads/Examensarbete/Data/TreesTowerFoot240829.gpkg"
out_file <- "C:/Users/digit/Downloads/Examensarbete/Results/ff3d_matched_trees.gpkg"

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

# ensure reference is projected
if (st_is_longlat(ref)) {
  stop("Reference file must be in projected coordinates (meters)")
}

# -----------------------------
# COMPUTE LAS TREE CENTROIDS
# -----------------------------
lasFF3D_df <- as.data.frame(lasFF3D@data)
lasCHM_df <- as.data.frame(lasCHM@data)

tree_centroids <- las_df %>%
  group_by(instance_pred) %>%
  summarise(
    x = mean(X),
    y = mean(Y)
  )

tree_centroids_sf <- st_as_sf(tree_centroids, coords = c("x","y"), crs = st_crs(ref))


# -----------------------------
# COMPUTE TREE HEIGHTS
# -----------------------------
tree_heights <- las_df %>%
  group_by(instance_pred) %>%
  arrange(desc(Z)) %>%
  slice_head(n = 1) %>%
  summarise(height = mean(Z))

# merge centroid + height
las_trees <- tree_centroids %>%
  left_join(tree_heights, by = "instance_pred")

las_trees_sf <- st_as_sf(las_trees, coords = c("x","y"), crs = st_crs(ref))

# -----------------------------
# FIND NEAREST LAS TREE FOR EACH REF TREE
# -----------------------------
nearest_index <- st_nearest_feature(ref, las_trees_sf)

nearest_dist <- st_distance(ref, las_trees_sf[nearest_index,], by_element = TRUE)
nearest_dist <- as.numeric(nearest_dist)

# -----------------------------
# APPLY DISTANCE FILTER
# -----------------------------
matched <- nearest_dist <= max_dist

ref$matched <- as.integer(matched)
ref$matched_dist <- nearest_dist
ref$matched_id <- NA
ref$matched_height <- NA

ref$matched_id[matched] <- las_trees$instance_pred[nearest_index[matched]]

# -----------------------------
# COMPUTE HEIGHT RATIO
# -----------------------------
ref$matched_height[matched] <-
  las_trees$height[nearest_index[matched]] / ref$H_TLS[matched]

# -----------------------------
# SAVE OUTPUT
# -----------------------------
st_write(ref, out_file, delete_dsn = TRUE)

print("Matching complete.")

# -----------------------------
# Caluclate Metrics
# -----------------------------

#Calculate TP, FN
matched_refs <- unique(seg_sf$ref_id_nn[seg_sf$matched])
fn_sf <- ref_sf[!(seq_len(nrow(ref_sf)) %in% matched_refs), ]
tp_sf <- ref_sf[(seq_len(nrow(ref_sf)) %in% matched_refs), ]

TP <- nrow(tp_sf)
TP
FN <- nrow(fn_sf)
FN


# -----------------------------
# Plot TP, FN
# -----------------------------

seg_sf %>%
  filter(matched == TRUE) %>%                #  ta bara omatchade
  count(ref_id_nn, sort = TRUE) %>%           # räkna förekomster
  head(10)  


##Plotting false matches and GT positions
true_sf <- ref[ref$matched == TRUE, ]
false_sf <- ref[ref$matched == FALSE, ]
chm <- rasterize_canopy(las = nlas, res = 0.5, algorithm = p2r(0.15))
plot(chm, col = col)
points(st_coordinates(tree_centroids_sf), col = "black", add = TRUE, cex = 0.5)
points(st_coordinates(true_sf), col = "purple", add = TRUE, cex = 0.7)
points(st_coordinates(false_sf), col = "black", add = TRUE, cex = 0.7)
points(st_coordinates(ref), col = "black", add = TRUE, cex = 0.4)

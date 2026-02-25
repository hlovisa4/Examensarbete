library(lidR)
library(ggplot2)
library(sf)


las_lidr <- readLAS("C:/Users/digit/Downloads/Examensarbete/Results/lidr_segmentation.las")
las_ff3d <- readLAS("C:/Users/digit/Downloads/Examensarbete/Results/")
ref_sf <- st_read("C:/Users/digit/Downloads/Examensarbete/Data/TreesTowerFoot240829.gpkg", quiet = TRUE)

metrics_lidr <- crown_metrics(las_lidr, ~list(z_max = max(Z), z_mean = mean(Z), x_mean = mean(X), y_mean = mean(Y)))
metrics_ff3d <- crown_metrics(las_ff3d, ~list(z_max = max(Z), z_mean = mean(Z), x_mean = mean(X), y_mean = mean(Y)))

max_dist <- 3

seg_sf <- st_as_sf(metrics_lidr, coords = c("x_mean","y_mean"), crs = st_crs(3006))
idx <- st_nearest_feature(seg_sf, ref_sf)
seg_sf$ref_id_nn <- ref_sf$GlobalID[idx]

# distance to its nearest reference point
seg_sf$dist_m <- as.numeric(st_distance(seg_sf, ref_sf[idx, ], by_element = TRUE))

# consider it a valid match only if within threshold
seg_sf$matched <- seg_sf$dist_m <= max_dist
seg_sf$ref_id_nn[!seg_sf$matched] <- NA

TP <- sum(seg_sf$matched)
TP

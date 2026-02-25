library(lidR)
library(dplyr)
library(sf)


las_filt <- filter_poi(las_seg_dal, "instance_pred" != -1)
metrics <- crown_metrics(las_filt, ~list(z_max = max(Z), centroid_x = mean(X), centroid_y = mean(Y)))
seg_sf <- st_as_sf(metrics, coords = c("centroid_x","centroid_y"), crs = st_crs(3006))
ref_sf <- st_read("C:/Users/Lovisa/Downloads/TreesTowerFoot240829.gpkg", quiet = TRUE)

idx <- st_nearest_feature(seg_sf, ref_sf)


###Setup input data
lidr_las <- readLAS()


ref_sf <- st_read("C:/Users/Lovisa/Downloads/TreesTowerFoot240829.gpkg", quiet = TRUE)



###Get TP
max_dist <- 3

idx <- st_nearest_feature(seg_sf, ref_sf)
seg_sf$ref_id_nn <- idx

# distance to its nearest reference point
seg_sf$dist_m <- as.numeric(st_distance(seg_sf, ref_sf[idx, ], by_element = TRUE))

# consider it a valid match only if within threshold
seg_sf$matched <- seg_sf$dist_m <= max_dist

#Calculate TP, FN
matched_refs <- unique(seg_sf$ref_id_nn[seg_sf$matched])
fn_sf <- ref_sf[!(seq_len(nrow(ref_sf)) %in% matched_refs), ]
tp_sf <- ref_sf[(seq_len(nrow(ref_sf)) %in% matched_refs), ]

TP <- nrow(tp_sf)
TP
FN <- nrow(fn_sf)
FN



seg_sf %>%
  filter(matched == TRUE) %>%                # ta bara omatchade
  count(ref_id_nn, sort = TRUE) %>%           # räkna förekomster
  head(10)  


##Plotting false matches and GT positions
false_sf <- seg_sf[seg_sf$matched == FALSE, ]
plot(chm, col = col)
points(st_coordinates(seg_sf), col = "black", add = TRUE, cex = 0.5)
points(st_coordinates(false_sf), col = "red", add = TRUE, cex = 0.7)
points(st_coordinates(ref_sf), col = "yellow", add = TRUE, cex = 0.4)

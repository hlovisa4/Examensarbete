library(lidR)
library(sf)
library(dplyr)

# -----------------------------
# Read Files
# -----------------------------
las_file_ff3d_full <- "C:/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/full_ALS_clipped_to_reflist.las" 
las_file_ff3d_tile <- "C:/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/tile_ALS_clipped_to_reflist.las" 
las_file_ff3d_ref <- "C:/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/ALS_clipped_to_reflist_round1.las" 
las_file_lidr <- "C:/Users/digit/Downloads/Examensarbete/Results/chm_segmentation/chm_ALS_clipped_to_reflist.las"
ref_file <- "C:/Users/digit/Downloads/Examensarbete/Data/TreesTowerFoot240829.gpkg" 
out_file <- "C:/Users/digit/Downloads/Examensarbete/Results/matched_trees.gpkg"

max_dist <- 2   # maximum allowed distance (m)


mycsf <- csf(sloop_smooth = FALSE, class_threshold = 0.5, cloth_resolution = 0.5, time_step = 0.65)

ref <- st_read(ref_file)
ref <- ref[!duplicated(ref$GlobalID), ]

process_las <- function(las_path, ref, max_dist, prefix) {
  las <- readLAS(las_path)
  las_csf <- classify_ground(las, mycsf) 
  las <- normalize_height(las_csf, knnidw())
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
  n_instance <- nrow(centroids)
  # -----------------------------
  # Heights
  # -----------------------------
  heights <- df %>%
    group_by(instance_pred) %>%
    summarise(height = max(Z), .groups = "drop")
  
  trees <- left_join(centroids, heights, by = "instance_pred")
  trees_sf <- st_as_sf(trees, coords = c("x","y"), crs = st_crs(ref))
  neighbor_list <- st_is_within_distance(trees_sf, trees_sf, dist = max_dist) 
  trees$neighbor_count <- lengths(neighbor_list)
  
  # -----------------------------
  # Matching
  # -----------------------------
  nn_idx <- st_nearest_feature(ref, trees_sf)
  nn_dist <- as.numeric(st_distance(ref, trees_sf[nn_idx,], by_element = TRUE))
  
  matched <- nn_dist <= max_dist
  tmp <- data.frame(
    ref_idx = seq_along(nn_idx),
    tree_idx = nn_idx,
    dist = nn_dist,
    matched = matched
  )
  tmp_matched <- tmp[tmp$matched, ]
  
  # For each detected tree, keep the closest reference
  keep_idx <- tmp_matched %>%
    group_by(tree_idx) %>%
    slice_min(order_by = dist, n = 1, with_ties = FALSE) %>%
    pull(ref_idx)
  
  # Reset matched: only keep best matches
  matched[] <- FALSE
  matched[keep_idx] <- TRUE
  # -----------------------------
  # Return as dataframe
  # -----------------------------
  out <- data.frame(
    matched = as.integer(matched),
    dist = nn_dist,
    id = NA,
    height = NA,
    height_ratio = NA,
    neighbors = NA
  )
  
  out$id[matched] <- trees$instance_pred[nn_idx[matched]]
  out$height[matched] <- trees$height[nn_idx[matched]]
  out$height_ratio[matched] <- trees$height[nn_idx[matched]] / ref$H_TLS[matched]
  out$neighbors[matched] <- trees$neighbor_count[nn_idx[matched]]
  precision <- sum(out$matched) / n_instance
  # add prefix to column names
  colnames(out) <- paste0(prefix, "_", colnames(out))
  
  return(list(precision = precision, out = out, n_instance = n_instance))
}


res_full <- process_las(las_file_ff3d_full, ref, max_dist, "ff3d_full")
ff3d_res_full <- res_full$out
res_full$precision

res_tile <- process_las(las_file_ff3d_tile, ref, max_dist, "ff3d_tile")
ff3d_res_tile <- res_tile$out
res_tile$precision


res_chm <- process_las(las_file_lidr, ref, max_dist, "chm")
chm_res <- res_chm$out
res_chm$precision


ref <- bind_cols(
  ref,
  ff3d_res_full,
  ff3d_res_tile,
  chm_res
)
neighbors_list <- st_is_within_distance(ref, ref, dist = 5)
ref$n_neighbors_10m <- lengths(neighbors_list) - 1

st_write(ref, out_file, delete_dsn = TRUE)

ref <- ref %>%
  mutate(neighbor_bin = cut(
    n_neighbors_10m,
    breaks = quantile(n_neighbors_10m, probs = seq(0, 1, 0.25), na.rm = TRUE),
    include.lowest = TRUE,
    labels = c("Low", "Medium", "High", "Very high")
  ))

analysis <- ref %>%
  group_by(neighbor_bin) %>%
  summarise(
    n = n(),
    ff3d_tile_success = mean(ff3d_tile_matched, na.rm = TRUE),
    chm_success = mean(chm_matched, na.rm = TRUE)
  )
library(tidyr)
library(ggplot2)

analysis_long <- analysis %>%
  pivot_longer(
    cols = c(ff3d_tile_success, chm_success),
    names_to = "method",
    values_to = "success_rate"
  )

ggplot(analysis_long, aes(x = neighbor_bin, y = success_rate, color = method, group = method)) +
  geom_line() +
  geom_point(size = 2) +
  labs(
    x = "Neighbors within 5 m",
    y = "Detection success rate",
    color = "Method"
  ) +
  theme_minimal()

ggplot(ref, aes(x = n_neighbors_10m, y = ff3d_tile_matched)) +
  geom_jitter(height = 0.02, alpha = 0.3) +
  geom_smooth(method = "glm", method.args = list(family = "binomial")) +
  theme_minimal()


ref_clean <- ref %>%
  filter(
    !is.na(ff3d_tile_matched),
    !is.na(n_neighbors_10m),
    !is.na(H_TLS)
  )
ref_clean$Species <- as.factor(ref_clean$Species)
model_null <- glm(ff3d_tile_matched ~ H_TLS,
                  data = ref_clean, family = binomial)

model_full <- glm(ff3d_tile_matched ~ H_TLS + n_neighbors_10m,
                  data = ref_clean, family = binomial)

anova(model_null, model_full, test = "Chisq")


#Interaction Model
model_all <- glm(ff3d_tile_matched ~ H_TLS + n_neighbors_10m + DBH_Field + Species, 
                  data = ref_clean, family = binomial)
model_base <- glm(ff3d_tile_matched ~ H_TLS +DBH_Field + Species, 
                 data = ref_clean, family = binomial)
anova(model_base, model_all, test = "Chisq")
summary(model_all)

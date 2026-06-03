library(lidR)
library(sf)
library(dplyr)
library(nngeo)
library(ggplot2)
library(concaveman)

# -----------------------------
# Read Files
# -----------------------------
las_file_ff3d_full <- "C:/Users/digit/Downloads/Examensarbete/Results/ff3d_segmentation/tile_ALS_clipped_to_reflist.las"
ref_file <- "C:/Users/digit/Downloads/Examensarbete/Data/TreesTowerFoot240829.gpkg"

ref <- st_read(ref_file, quiet = TRUE)

# -----------------------------
# Parameters
# -----------------------------
plot_center_x <- 731337.678
plot_center_y <- 7134016.31

plot_radius <- 20  # meters
max_dist <- 1.8

# -----------------------------
# Read and normalize LAS
# -----------------------------
mycsf <- csf(
  class_threshold = 0.5,
  cloth_resolution = 0.5,
  time_step = 0.65
)

las <- readLAS(las_file_ff3d_full)

if (is.empty(las)) stop("LAS file is empty.")

las_csf <- classify_ground(las, mycsf)
las_norm <- normalize_height(las_csf, knnidw())

df <- as.data.frame(las_norm@data)

# -----------------------------
# Clip subsection (circle)
# -----------------------------
df_sub <- df %>%
  mutate(
    dist = sqrt((X - plot_center_x)^2 + (Y - plot_center_y)^2)
  ) %>%
  filter(dist <= plot_radius)
df_sub <- df_sub %>%
  filter(!is.na(instance_pred),
         instance_pred > 0)
# -----------------------------
# Centroids
# -----------------------------
centroids <- df_sub %>%
  filter(!is.na(instance_pred)) %>%
  group_by(instance_pred) %>%
  summarise(
    x = mean(X),
    y = mean(Y),
    height = max(Z),
    .groups = "drop"
  )

centroids_sf <- st_as_sf(
  centroids,
  coords = c("x", "y"),
  crs = st_crs(ref)
)

tree_tops <- df_sub %>%
  group_by(instance_pred) %>%
  slice_max(order_by = Z, n = 1, with_ties = FALSE) %>%
  ungroup()

tree_tops_sf <- st_as_sf(
  tree_tops,
  coords = c("X", "Y"),
  crs = st_crs(ref)
)

# -----------------------------
# Reference trees in subsection
# -----------------------------
plot_circle <- st_buffer(
  st_sfc(
    st_point(c(plot_center_x, plot_center_y)),
    crs = st_crs(ref)
  ),
  dist = plot_radius
)

ref_sub <- st_intersection(ref, plot_circle)

# -----------------------------
# Crown polygons
# -----------------------------
polygon_list <- list()

instance_ids <- unique(df_sub$instance_pred)
instance_ids <- instance_ids[!is.na(instance_ids)]

for(id in instance_ids){
  
  pts <- df_sub %>%
    filter(instance_pred == id)
  
  # skip tiny segments
  if(nrow(pts) < 20) next
  
  # create sf points object
  pts_sf <- st_as_sf(
    pts,
    coords = c("X", "Y"),
    crs = st_crs(ref)
  )
  
  # concave hull
  crown_poly <- concaveman(pts_sf)
  
  # skip invalid geometries
  if(length(crown_poly) == 0) next
  
  crown_poly <- st_sf(
    instance_pred = id,
    geometry = st_geometry(crown_poly)
  )
  
  polygon_list[[length(polygon_list) + 1]] <- crown_poly
}

crowns_sf <- do.call(rbind, polygon_list)
# -----------------------------
# Match reference trees to polygons
# -----------------------------

# Ensure valid geometries
crowns_sf <- st_make_valid(crowns_sf)

# Spatial join:
# adds polygon attributes to reference trees
ref_matches <- st_join(
  ref_sub,
  crowns_sf,
  join = st_within,
  left = TRUE
)

# Trees inside a segmented polygon
ref_inside <- ref_matches %>%
  filter(!is.na(instance_pred))

# Trees NOT inside any polygon
ref_outside <- ref_matches %>%
  filter(is.na(instance_pred))

# Counts
n_total_ref <- nrow(ref_sub)
n_inside <- nrow(ref_inside)
n_outside <- nrow(ref_outside)

cat("\n--- Segmentation Coverage ---\n")
cat("Total reference trees:", n_total_ref, "\n")
cat("Inside segmented polygon:", n_inside, "\n")
cat("Outside segmented polygon:", n_outside, "\n")
cat("Coverage (%):", round(100 * n_inside / n_total_ref, 1), "\n")

# -----------------------------
# Optional:
# How many reference trees per segment
# -----------------------------

segment_stats <- ref_inside %>%
  st_drop_geometry() %>%
  group_by(instance_pred) %>%
  summarise(
    n_reference_trees = n(),
    .groups = "drop"
  )

print(segment_stats)

# Segments containing multiple reference trees
multi_match_segments <- segment_stats %>%
  filter(n_reference_trees > 1)

cat("\nSegments with multiple reference trees:",
    nrow(multi_match_segments), "\n")

# -----------------------------
# Plot
# -----------------------------
fig <- ggplot() +
  
  # Crown polygons
  geom_sf(
    data = crowns_sf,
    aes(fill = as.factor(instance_pred)),
    color = "black",
    alpha = 0.4,
    linewidth = 0.3,
    show.legend = FALSE
  ) +
  
  # Instance centroids
  geom_sf(
    data = centroids_sf,
    color = "red",
    size = 2
  ) +
  
  geom_sf(
    data = tree_tops_sf,
    color = "darkgreen",
    shape = 17,   # triangle
    size = 2.5
  ) +
  
  # Reference trees
  geom_sf(
    data = ref_sub,
    color = "blue",
    size = 2,
    shape = 4,
    stroke = 1.2
  ) +
  
  # Plot boundary
  geom_sf(
    data = plot_circle,
    fill = NA,
    color = "black",
    linewidth = 0.8,
    linetype = "dashed"
  ) +
  
  coord_sf() +
  
  labs(
    title = "Tree Instance Segmentation Subsection",
    subtitle = "Black polygons = instance outlines, Red = centroids, Blue = reference trees"
  ) +
  
  theme_minimal() +
  
  theme(
    panel.grid.major = element_line(color = "grey85"),
    plot.title = element_text(face = "bold")
  )
ggsave("C:/Users/digit/Downloads/Examensarbete/Examensarbete/4.seg_evaluation/test_segmentation_overview.png", fig, width = 10.5, height = 8.0, dpi = 400, bg = "white")

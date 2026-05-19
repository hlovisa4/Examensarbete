library(sf)
library(dplyr)
library(ggplot2)

# -----------------------------
# Read data
# -----------------------------
trees <- st_read("C:/Users/digit/Downloads/Examensarbete/Data/TreesTowerFoot240829.gpkg")

# Ensure projected CRS in meters
trees <- st_transform(trees, 3006)

# -----------------------------
# Distance matrix
# -----------------------------
d <- st_distance(trees)

# Ignore self-distance
diag(d) <- Inf

# -----------------------------
# Nearest neighbour distance
# -----------------------------
trees$nearest_dist <- apply(d, 1, min)

# -----------------------------
# Number of neighbours
# -----------------------------
# Define neighborhood radius (meters)
radius <- 5

# Count neighbors within radius
trees$n_neighbors <- apply(d, 1, function(x) sum(x <= radius))

# -----------------------------
# Species-wise summaries
# -----------------------------
# Replace "Species" with your actual species column name
species_summary <- trees %>%
  st_drop_geometry() %>%
  group_by(Species) %>%
  summarise(
    n_trees = n(),
    mean_height = mean(H_TLS, na.rm = TRUE),
    sd_height = sd(H_TLS, na.rm = TRUE),
    
    mean_nearest_dist = mean(nearest_dist, na.rm = TRUE),
    median_nearest_dist = median(nearest_dist, na.rm = TRUE),
    
    mean_neighbors = mean(n_neighbors, na.rm = TRUE),
    median_neighbors = median(n_neighbors, na.rm = TRUE)
  )

print(species_summary)

# -----------------------------
# Height distribution per species
# -----------------------------
ggplot(
  st_drop_geometry(trees),
  aes(x = H_TLS)
) +
  geom_histogram(bins = 30) +
  facet_wrap(~ Species, scales = "free_y") +
  theme_bw() +
  labs(
    title = "Height distribution per species",
    x = "Tree height (m)",
    y = "Count"
  )

# -----------------------------
# Nearest-neighbour distance distribution
# -----------------------------
ggplot(
  st_drop_geometry(trees),
  aes(x = nearest_dist)
) +
  geom_histogram(bins = 30) +
  facet_wrap(~ Species, scales = "free_y") +
  theme_bw() +
  labs(
    title = "Nearest-neighbour distance per species",
    x = "Nearest neighbour distance (m)",
    y = "Count"
  )

# -----------------------------
# Mean neighbor count per species
# -----------------------------
ggplot(
  species_summary,
  aes(x = reorder(Species, mean_neighbors),
      y = mean_neighbors)
) +
  geom_col() +
  coord_flip() +
  theme_bw() +
  labs(
    title = paste("Mean number of neighbours within", radius, "m"),
    x = "Species",
    y = "Mean neighbour count"
  )

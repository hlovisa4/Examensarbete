library(lidR)
library(ggplot2)
library(RCSF)

#Inspect the data
las <- readLAS("C:/Users/digit/Downloads/Examensarbete/Data/las_polygon/240829_ALS_Matrice300_Svb_clipped.las",  filter = "-set_withheld_flag 0 -drop_z_above 297")

col <- height.colors(50)
print(las)
las_check(las)


#plot(las)


#Ground classification
mycsf <- csf(sloop_smooth = FALSE, class_threshold = 0.5, cloth_resolution = 0.5, time_step = 0.65)
las_csf <- classify_ground(las1, mycsf)
#plot(las_csf, color = "Classification", size = 3, bg = "white") 
#visualise a cross section to evaluate the classification:
p1 <- c(731300, 7134000)
p2 <- c(731310, 7134050)
las_tr <- clip_transect(las_csf, p1, p2, width = 5, xz = TRUE)

ggplot(payload(las_tr), aes(X,Z, color = Classification)) + 
  geom_point(size = 0.5) + 
  coord_equal() + 
  theme_minimal() +
  scale_color_gradientn(colours = height.colors(50))

##Create a digital terrain model
#dtm_tin <- rasterize_terrain(las_csf, res = 1, algorithm = tin())
#plot_dtm3d(dtm_tin, bg = "white") 

#Height normalisation
nlas <- normalize_height(las_csf, knnidw())
#ändra algoritm
hist(filter_ground(nlas)$Z, breaks = seq(-0.5, 0.5, 0.01), main = "", xlab = "Elevation")



# Generatelas = # Generate CHM
chm <- rasterize_canopy(las = nlas, res = 0.5, algorithm = p2r(0.15))
plot(chm, col = col)

# Generate kernel and smooth chm
kernel <- matrix(1, 3, 3)
schm <- terra::focal(x = chm, w = kernel, fun = mean, na.rm = TRUE)
plot(schm, col = col)

# Detect trees
f <- function(x) {y <- 2.6 * (-(exp(-0.08*(x-2))-1)) + 1.5
  y[x < 2] <- 1.5
  y[x > 20] <- 5
  return(y)}

ttops <- locate_trees(las = schm, algorithm = lmf(f))
ttops

plot(ttops, col = "black", add = TRUE, cex = 0.5)

# Segment trees using dalponte
#las_seg <- segment_trees(las = nlas, algorithm = li2012(dt1=1.4) )
las_seg <- segment_trees(las = nlas, algorithm = dalponte2016(chm = schm, treetops = ttops) )
# Count number of trees detected and segmented
length(unique(las_seg$treeID) |> na.omit())

las_seg <- add_lasattribute(
  las_seg,
  las_seg$treeID,
  name = "instance_pred",
  desc = "Predicted instance ID")

# Check
names(las_seg)
las_seg$instance_pred[is.na(las_seg$instance_pred)] <- -1
unique(las_seg$instance_pred) |> head()
plot(las_seg, color = "instance_pred")

las_final <- unnormalize_height(las_seg)

writeLAS(las_final, file= "C:/Users/digit/Downloads/Examensarbete/Results/lidr_segmentation.las" )





#Extract positions:
library(dplyr)

trees_xyh <- las_seg@data %>%
  filter(!is.na(treeID)) %>%
  group_by(treeID) %>%
  summarise(
    x = median(X),
    y = median(Y),
    z_max = max(Z),
    z_med = median(Z),
    n_points = n(),
    .groups = "drop"
  ) %>%
filter(z_max <= 30) 

hist(trees_xyh$z_max,
     breaks = 40,
     col = "forestgreen",
     main = "Histogram of Tree Heights, segmentation",
     xlab = "Tree height (m)",
     ylab = "Number of trees")


#Match fo treefile:
library(sf)
library(dplyr)

seg_sf <- st_as_sf(trees_xyh, coords = c("x","y"), crs = st_crs(3006))  # <-- set your CRS!
ref_sf <- st_read("C:/Users/Lovisa/Downloads/TreesTowerFoot240829.gpkg", quiet = TRUE)
ref_sf
hist(ref_sf$H_TLS,
     breaks = 40,
     col = "forestgreen",
     main = "Histogram of Tree Heights, ref",
     xlab = "Tree height (m)",
     ylab = "Number of trees")
plot(chm, col = hcl.colors(60, "YlGnBu"), main = "CHM + reference trees")
points(st_coordinates(ref_sf), col = "black", add = TRUE, cex = 0.5)

########
max_dist <- 15

idx <- st_nearest_feature(seg_sf, ref_sf)
seg_sf$ref_id_nn <- ref_sf$ref_id[idx]

# distance to its nearest reference point
seg_sf$dist_m <- as.numeric(st_distance(seg_sf, ref_sf[idx, ], by_element = TRUE))

# consider it a valid match only if within threshold
seg_sf$matched <- seg_sf$dist_m <= max_dist
seg_sf$ref_id_nn[!seg_sf$matched] <- NA

TP <- sum(seg_sf$matched)
TP

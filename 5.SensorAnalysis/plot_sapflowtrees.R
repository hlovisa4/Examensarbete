library(sf)
library(ggplot2)
library(RCSF)
library(viridis) 
library(patchwork) 


coords <- data.frame(
  x = c(731341.472, 731337.678, 
         731366.628, 731361.967),
  y = c(7134019.67, 7134016.31, 
        7134011.897, 7134010.787)
)

pts <- st_as_sf(coords, coords = c("x", "y"), crs = 3006)  # likely SWEREF99 TM

las <- readLAS("C:/Users/digit/Downloads/Examensarbete/Data/las_polygon/240829_ALS_Matrice300_Svb_clipped.las",  filter = "-set_withheld_flag 0 -drop_z_above 297")
mycsf <- csf(sloop_smooth = FALSE, class_threshold = 0.5, cloth_resolution = 0.5, time_step = 0.65)
las_csf <- classify_ground(las, mycsf)
nlas <- normalize_height(las_csf, knnidw())
chm <- rasterize_canopy(las = nlas, res = 0.5, algorithm = p2r(0.15))


library(ggplot2)

chm_df <- as.data.frame(chm, xy = TRUE)

plot(chm, col = height.colors(50))
plot(pts, add = TRUE, col = "red", pch = 16, cex = 1.2)

coords <- st_coordinates(pts)          # extract X/Y from geometry
pts <- st_drop_geometry(pts)
pts$x <- coords[, "X"]
pts$y <- coords[, "Y"]

df_chm <- as.data.frame(chm, xy = TRUE, na.rm = TRUE)
names(df_chm) <- c("x", "y", "z")
fig <- ggplot(df_chm, aes(x, y, fill = z)) +
  geom_raster() +
  geom_point(data =pts, aes(x, y), inherit.aes = FALSE,
             shape = 4, size = 1.4, stroke = 1, fill = "black") +
  coord_equal() +
  scale_fill_viridis(name = "Height (m)", option = "C") +
  labs(title = "Trees with sap flow sensors", x = "X (m)", y = "Y (m)")
ggsave("C:/Users/digit/Downloads/Examensarbete/Results/Sapflow_tree_plot.png", fig, width = 10.5, height = 8.0, dpi = 400, bg = "white")

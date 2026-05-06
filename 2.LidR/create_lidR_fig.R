library(lidR)
library(ggplot2)
library(terra)
library(patchwork)   # install.packages("patchwork")
library(viridis)     # install.packages("viridis")
library(sf)
library(rgl)
library(png)
library(grid)

# ----------------------------
# 0) Reproducible settings
# ----------------------------
theme_set(theme_minimal(base_size = 12))

out_png <- "C:/Users/digit/Downloads/Examensarbete/Results/figure_lidr_pipeline.png"

panel_theme <- theme(
  plot.title = element_text(face = "bold", hjust = 0.5),
  plot.margin = margin(t = 10, r = 10, b = 10, l = 10)
)
las <- readLAS("C:/Users/digit/Downloads/Examensarbete/Data/ALS_clipped_to_reflist.las",  filter = "-set_withheld_flag 0 -drop_z_above 297")
mycsf <- csf(sloop_smooth = FALSE, class_threshold = 0.5, cloth_resolution = 0.5, time_step = 0.65)
las_csf <- classify_ground(las, mycsf)
nlas <- normalize_height(las_csf, knnidw())
chm <- rasterize_canopy(las = nlas, res = 0.5, algorithm = p2r(0.15))
kernel <- matrix(1, 3, 3)
schm <- terra::focal(x = chm, w = kernel, fun = mean, na.rm = TRUE)
f <- function(x) {y <- 2.6 * (-(exp(-0.08*(x-2))-1)) + 1.5
y[x < 2] <- 1.5
y[x > 20] <- 5
return(y)}

ttops <- locate_trees(las = schm, algorithm = lmf(f))
las_seg <- segment_trees(las = nlas, algorithm = dalponte2016(chm = schm, treetops = ttops) )

# ----------------------------
# 1) Panel A: ground classification cross-section
# ----------------------------
# Use your transect (las_tr) if you already made it. Otherwise:
p1 <- c(731300, 7134000)
p2 <- c(731310, 7134050)
las_tr <- clip_transect(las_csf, p1, p2, width = 5, xz = TRUE)

df_tr <- payload(las_tr)


pA <- ggplot(df_tr, aes(X, Z, color = factor(Classification))) +
  geom_point(size = 0.35, alpha = 0.8) +
  coord_equal() +
  scale_color_manual(
    values = c(
      "2" = "#8c510a",  # ground
      "0" = "#1b7837"  # non-ground
    ),
    labels = c(
      "2" = "Ground",
      "0" = "Non-ground"
    )
  ) +
  labs(
    title = "A) Ground classification (cross-section)",
    x = "X (m)", y = "Z (m)", color = "Class"
  ) +
  panel_theme
  

# ----------------------------
# 2) Panel B: CHM
# ----------------------------
df_chm <- as.data.frame(chm, xy = TRUE, na.rm = TRUE)
names(df_chm) <- c("x", "y", "z")

pB <- ggplot(df_chm, aes(x, y, fill = z)) +
  geom_raster() +
  coord_equal() +
  scale_fill_viridis(name = "Height (m)", option = "C") +
  labs(title = "B) Canopy height model (CHM)", x = "X (m)", y = "Y (m)") +
  panel_theme

# ----------------------------
# 3) Panel C: Smoothed CHM + treetops
# ----------------------------
df_schm <- as.data.frame(schm, xy = TRUE, na.rm = TRUE)
names(df_schm) <- c("x", "y", "z")

tt_sf <- st_as_sf(ttops)                 # make sure it's sf
coords <- st_coordinates(tt_sf)          # extract X/Y from geometry
tt <- st_drop_geometry(tt_sf)
tt$x <- coords[, "X"]
tt$y <- coords[, "Y"]

pC <- ggplot(df_schm, aes(x, y, fill = z)) +
  geom_raster() +
  geom_point(data = tt, aes(x, y), inherit.aes = FALSE,
             shape = 4, size = 1.4, stroke = 0.4, fill = "black") +
  coord_equal() +
  scale_fill_viridis(name = "Height (m)", option = "C") +
  labs(title = "C) Smoothed CHM + detected treetops", x = "X (m)", y = "Y (m)") +
  panel_theme

# ----------------------------
# 4) Panel D: Segmentation as a raster (treeID / instance_pred)
#    This is usually the cleanest "map view" for a thesis.
# ----------------------------
# Use instance_pred if you created it; otherwise use treeID.
id_name <- if ("instance_pred" %in% names(las_seg)) "instance_pred" else "treeID"

# ---- 4.1 Make a zoomed LAS for a readable crown-scale view ----
las_zoom <- clip_circle(las_seg, x = 731300, y = 7134050, radius = 24)

# ---- 4.2 Render high-resolution rgl snapshot ----
rgl_file <- "C:/Users/digit/Downloads/Examensarbete/Results/panelD_rgl.png"

plot(las_zoom, color = id_name, size = 3, bg = "white", )
par3d(windowRect = c(50, 50, 3500, 3500))
rgl.snapshot(rgl_file)

close3d()

# ---- 4.3 Read the image and convert to a ggplot panel ----
img <- readPNG(rgl_file)

pD <- ggplot() +
  annotation_custom(
    rasterGrob(img, interpolate = TRUE),
    xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf
  ) +
  coord_fixed() +
  labs(title = "D) Tree instance segmentation (point cloud view)") +
  theme_void() +
  panel_theme
# ----------------------------
# 5) Combine + export
# ----------------------------
pA <- pA + theme(aspect.ratio = 1)
pB <- pB + theme(aspect.ratio = 1)
pC <- pC + theme(aspect.ratio = 1)
pD <- pD + theme(aspect.ratio = 1)
fig <- (pA | pB) / (pC | pD) +
  plot_layout(
    widths = c(1, 1.4),   # left column narrower than right
    heights = c(1, 1.2)   # top row shorter than bottom
  ) +
  plot_annotation(
    title = "Tree segmentation workflow in lidR",
    theme = theme(plot.title = element_text(face = "bold", size = 14))
  )

ggsave(out_png, fig, width = 10.5, height = 8.0, dpi = 400, bg = "white")
out_png

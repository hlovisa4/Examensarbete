library(lidR)
library(ggplot2)
library(sf)

las <-  readLAS("C:/Users/digit/Downloads/Examensarbete/Results/TLS_labeled_from_ALS_lidr.las")
#Plotta trädtopparna

las <- add_lasattribute(las, las$instance_pred, name = "treeID", desc = "Tree instance id (copied from instance_pred)")

inst_34 <- filter_poi(las, instance_pred == 130)
plot(inst_34)

writeLAS(inst_34, file= "C:/Users/digit/Downloads/Examensarbete/Results/instance_130.las")

metrics <- crown_metrics(las, ~list(z_max = max(Z), z_mean = mean(Z), x_mean = mean(X), y_mean = mean(Y)))

plot(chm, col = col)
points(
  metrics$x_mean,
  metrics$y_mean,
  pch = 19,
  cex = 0.6,
  col = "black"
)


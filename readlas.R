library(lidR)
library(RCSF)

las <- readLAS("C:/Users/digit/Downloads/Examensarbete/Results/TLS_labeled_from_ALS_lidr.las") 
print(las)
length(unique(las$instance_pred))

plot(las, color = "instance_pred")
flas <-  filter_poi(las, instance_pred !=-1)
plot(flas, color = "semantic_pred")
plot(filter_poi(las, instance_pred == 0), color = "instance_pred")

las <- add_attribute(las, las$instance_pred, "treeID")

metrics <- crown_metrics(las, ~list(z_max = max(Z), z_mean = mean(Z))) # calculate tree metrics

plot(chm, col = col)
plot(metrics["z_max"], pal = hcl.colors, pch = 10, add = TRUE) # plot using z_max


las_11_113<-  filter_poi(las, instance_pred == c("44", "45"))
metrics_11_113 <- crown_metrics(las_11_113, ~list(z_max = max(Z), z_mean = mean(Z)))
plot(chm, col = col)
plot(metrics_11_113["z_max"], pal = hcl.colors, pch = 10, add = TRUE) # plot using z_max

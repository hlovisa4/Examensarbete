library(lidR)

las <- readLAS("C:/Users/digit/Downloads/Examensarbete/Results/TLS_labeled_from_ALS_lidr_260304_thin.las")

filtlas <- filter_poi(las, instance_pred == -1)

##Plotta histogram över instance_pred
hist(filtlas$Z, 
     main = "Distribution of Z-values (instance_pred == -2)", 
     xlab = "Elevation (m)", 
     col = "steelblue", 
     breaks = 50)

library(dplyr)
library(ggplot2)



# 1. Convert LAS data to dataframes
df_total <- as.data.frame(las@data)
df_filt  <- as.data.frame(filtlas@data)

# 2. Define bin size (e.g., 0.5 meter intervals)
bin_size <- 0.5

# 3. Process total points
dist_total <- df_total %>%
  mutate(Z_bin = round(Z / bin_size) * bin_size) %>%
  group_by(Z_bin) %>%
  summarise(count_total = n())

# 4. Process filtered points
dist_filt <- df_filt %>%
  mutate(Z_bin = round(Z / bin_size) * bin_size) %>%
  group_by(Z_bin) %>%
  summarise(count_filt = n())

# 5. Merge and calculate percentage
comparison <- full_join(dist_total, dist_filt, by = "Z_bin") %>%
  mutate(count_filt = ifelse(is.na(count_filt), 0, count_filt),
         percentage = (count_filt / count_total) * 100)

ggplot(comparison, aes(x = Z_bin, y = percentage)) +
  geom_line(color = "firebrick", size = 1) +
  geom_area(fill = "firebrick", alpha = 0.2) +
  theme_minimal() +
  labs(title = "Percentage of Points with instance_pred == -2 by Elevation",
       x = "Elevation (Z)",
       y = "Percentage of Total Points (%)")

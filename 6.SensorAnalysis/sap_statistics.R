install.packages(c("tidyverse", "lme4", "lmerTest", "emmeans", "ggeffects"))
library(tidyverse)
library(lme4)
library(lmerTest)   # p-values
library(emmeans)    # post-hoc
library(ggeffects)  # predictions for plotting

df <- read.csv("C:/Users/Lovisa/Downloads/Examensarbete/5.SensorAnalysis/sapanalysis.csv")

# Convert to correct types
df <- df %>%
  mutate(
    treeid = as.factor(tree),
    species = as.factor(species), #,aspect = as.factor(aspect),   #North/South
    sensor = as.factor(sensor_type)
  )

biomass_long <- biomass_df %>%
  pivot_longer(
    cols = c(
      "0-1.3m Stem", "1.3-8m Stem", "8-14m Stem", "14+m Stem",
      "0-1.3m Foliage", "1.3-8m Foliage", "8-14m Foliage", "14+m Foliage"
    ),
    names_to = c("height_bin", "component"),
    names_sep = " "
  )

#plot difference biomass north/ south
biomass_df$diff_NS <- biomass_df$biomass_north - biomass_df$biomass_south
boxplot(biomass_df$diff_NS,
        main = "North - South biomass difference",
        ylab = "Biomass difference")
boxplot(diff_NS ~ height_bin, data = biomass_long)

#boxplot differences between sap flow north/south
sapflow_long <- sapflow_df %>%
  pivot_longer(cols = c("north", "south"),
               names_to = "side",
               values_to = "sap_flow")
boxplot(sap_flow ~ side, data = sapflow_long,
        main = "Sap flow: North vs South")

#T-test between biomass north / south
t.test(biomass_north, biomass_south, paired = TRUE)

#T-test between sap flow north/south
t.test(sap_flow ~ side, data = sapflow_long, paired = TRUE)

#Plot regression and R2 of biomass per height (aggregate north/south) and sap flow (aggregate north/south)
agg <- merge(biomass_long, sapflow_long, by = c("id", "height_bin"))
model <- lm(sap_flow ~ biomass + height_bin, data = agg)
summary(model)
lm_list <- lapply(unique(agg$height_bin), function(h) {
  lm(sap_flow ~ biomass, data = subset(agg, height_bin == h))
})
ggplot(agg, aes(x = biomass, y = sap_flow)) +
  geom_point() +
  geom_smooth(method = "lm") +
  facet_wrap(~height_bin)
summary(model)$r.squared

#Plot % of foliage biomass per bin
biomass_df <- biomass_df %>%
  mutate(across(ends_with("Foliage"),
                ~ .x / biomass_foliage,
                .names = "pct_{.col}"))
barplot(colMeans(biomass_df[, grep("pct", names(biomass_df))], na.rm=TRUE))

#Histogram of biomass per height bin vs sap flow at each height
ggplot(agg, aes(x = biomass)) +
  geom_histogram(bins = 30) +
  facet_wrap(~height_bin)

ggplot(agg, aes(x = biomass, y = sap_flow)) +
  geom_bin2d() +
  facet_wrap(~height_bin)


model <- lmer(WC ~ species + height + aspect + biomass + (1 | treeid), data = df)
summary(model)

model2 <- lmer(WC ~ species * height  + biomass * aspect + (1 | treeid), data = df)
summary(model2)

par(mfrow = c(1,2))
plot(model)              # residuals vs fitted
qqnorm(residuals(model))
qqline(residuals(model))

#Raw data overview
ggplot(df, aes(x = height, y = WC, color = species)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(~ species) +
  theme_minimal()

#Biomass effect
ggplot(df, aes(x = biomass, y = WC, color = species)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(~ species) +
  theme_minimal()

#Model prediction
pred <- ggpredict(model, terms = c("biomass"))

plot(pred) + theme_minimal()

#north/south difference
df_diff <- df %>%
  pivot_wider(names_from = aspect, values_from = WC) %>%
  mutate(diff_NS = N - S)

ggplot(df_diff, aes(x = biomass, y = diff_NS, color = species)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  theme_minimal()

#posthoc comparisons
emmeans(model, pairwise ~ treeid | species)

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

model <- lmer(WC ~  height * biomass + species+ (1 | treeid),  data = df)
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

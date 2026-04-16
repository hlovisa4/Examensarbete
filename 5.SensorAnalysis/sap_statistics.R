install.packages(c("tidyverse", "lme4", "lmerTest", "emmeans", "ggeffects"))
library(tidyverse)
library(lme4)
library(lmerTest)   # p-values
library(emmeans)    # post-hoc
library(ggeffects)  # predictions for plotting

df <- read.csv("your_file.csv")

# Convert to correct types
df <- df %>%
  mutate(
    treeid = as.factor(tree),
    species = as.factor(species),
    aspect = as.factor(aspect),   #North/South
  )

model <- lmer(WC ~ species + height + aspect + biomass + (1 | treeid), data = df)
summary(model)

model2 <- lmer(WC ~ species * height * aspect + biomass * aspect + (1 | treeid), data = df)
summary(model2)

par(mfrow = c(1,2))
plot(model2)              # residuals vs fitted
qqnorm(residuals(model2))
qqline(residuals(model2))

#Raw data overview
ggplot(df, aes(x = height, y = WC, color = aspect)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(~ species) +
  theme_minimal()

#Biomass effect
ggplot(df, aes(x = biomass, y = WC, color = aspect)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(~ species) +
  theme_minimal()

#Model prediction
pred <- ggpredict(model2, terms = c("biomass", "aspect", "species"))

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
emmeans(model2, pairwise ~ aspect | species)

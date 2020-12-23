# author: Junming Zhang
# build a multilevel linear regression model (RDD)
# and a normal MLR (simple linear without mixing)
# with the generated data with the train set and
# test against the test set

# all packages required for functionality
library(tidyverse)
library(lme4)
library(lmerTest)
library(knitr)
library(MuMIn)
library(broom)
library(broom.mixed)

# adjust output format of each table
# options(knitr.table.format = function() {
#   if (knitr::is_latex_output()) 
#     "latex" else "pipe"
# })

work_path = "/Users/peterzhang/Desktop/UTSG/STA/project/work"
setwd(work_path)

# load the data generated from datagen
train_set = read.csv("../data/train_set.csv")
test_set = read.csv("../data/test_set.csv")

# standardize the variables to scale the parameters
# so the model parameters are more useful and intuitive
standardize <- function(column) {
  result = (column - mean(column)) / sd(column)
  return(result)
}

# standardize the predictor and response variables in the train_set and test_set separately for two reasons:
# 1. the model will be resused, and the new data provided will be standardized separately as well, standardize two datasets
#    in one dataset might not help for diagnosis if we test against the test set
# 2. based on law of large number and central limit theorem, if the data set are large enough, the side effect caused by
#    standardization will be eliminated because the model parameters will converge to the expectation and normal distribution
train_set %>% mutate(std_new_cases_share_5_years_later = standardize(new_cases_5_years_later)) %>%
  mutate(std_hdi = standardize(hdi)) %>% mutate(std_haq = standardize(haq)) %>% mutate(std_gdp_per_capita = standardize(gdp_per_capita)) %>%
  mutate(std_female_emply_rate = standardize(female_emply_rate)) %>% mutate(std_urb_rate = standardize(urb_rate)) %>%
  mutate(std_drug_alcohol_disorder_share = standardize(drug_alcohol_disorder_share)) -> train_set

test_set %>% mutate(std_new_cases_share_5_years_later = standardize(new_cases_5_years_later)) %>%
  mutate(std_hdi = standardize(hdi)) %>% mutate(std_haq = standardize(haq)) %>% mutate(std_gdp_per_capita = standardize(gdp_per_capita)) %>%
  mutate(std_female_emply_rate = standardize(female_emply_rate)) %>% mutate(std_urb_rate = standardize(urb_rate)) %>%
  mutate(std_drug_alcohol_disorder_share = standardize(drug_alcohol_disorder_share)) -> test_set

# MLR with only 1 level
model_linear <- lm(std_new_cases_share_5_years_later ~ std_hdi + std_haq +
                   std_gdp_per_capita + std_female_emply_rate + std_urb_rate +
                   std_drug_alcohol_disorder_share + hdi_over_avg + haq_over_avg +
                   gdp_per_capita_over_avg + female_emply_rate_over_avg +
                   urb_rate_over_avg + drug_alcohol_disorder_share_over_avg, data = train_set)

# FREQUENTIST Random Intercept (multilevel) Model - Linear Regression
model_multilevel_linear <- lmer(std_new_cases_share_5_years_later ~ std_hdi + std_haq +
                       std_gdp_per_capita + std_female_emply_rate + std_urb_rate +
                       std_drug_alcohol_disorder_share + hdi_over_avg + haq_over_avg +
                       gdp_per_capita_over_avg + female_emply_rate_over_avg +
                       urb_rate_over_avg + drug_alcohol_disorder_share_over_avg + (1|Entity),
                     data = train_set, 
                     REML=F)


# model diagnosis on the normal MLR
kable(anova(model_linear), caption = "ANOVA table")
aic = AIC(model_linear)
bic = BIC(model_linear)
r_squared = summary(model_linear)$r.squared
AIC_BIC_R2_table = cbind(aic, bic, r_squared)
kable(AIC_BIC_R2_table, caption = "AIC & BIC measurement")

# compute correlation accuracy
distPred <- predict(model_linear, test_set)
# make actuals_predicteds dataframe
actuals_preds <- data.frame(cbind(actuals=test_set$std_new_cases_share_5_years_later, predicteds=distPred))
correlation_accuracy <- cor(actuals_preds)
kable(correlation_accuracy)
min_max_accuracy <- mean(apply(abs(actuals_preds), 1, min) / apply(abs(actuals_preds), 1, max))
mape <- abs(mean(abs((actuals_preds$predicteds - actuals_preds$actuals))/actuals_preds$actuals))
acc_err_table = cbind(min_max_accuracy, mape)
kable(acc_err_table, caption = "min max accuracy & mean absolute percentage error")

# a summary table on beta values of the simple MLR
model_linear %>% broom::tidy() %>% select(c("term", "estimate")) %>% kable()

# model diagnosis on the multilevel MLR
kable(anova(model_multilevel_linear), caption = "ANOVA table")
kable(r.squaredGLMM(model_multilevel_linear), caption = "R squared")
aic = AIC(model_multilevel_linear)
bic = BIC(model_multilevel_linear)
AIC_BIC_table = cbind(aic, bic)
kable(AIC_BIC_table, caption = "AIC & BIC measurement")

# compute correlation accuracy
distPred <- predict(model_multilevel_linear, test_set, allow.new.levels = TRUE)
# make actuals_predicteds dataframe
actuals_preds <- data.frame(cbind(actuals=test_set$std_new_cases_share_5_years_later, predicteds=distPred))
correlation_accuracy <- cor(actuals_preds)
kable(correlation_accuracy)
min_max_accuracy <- mean(apply(abs(actuals_preds), 1, min) / apply(abs(actuals_preds), 1, max))
mape <- abs(mean(abs((actuals_preds$predicteds - actuals_preds$actuals))/actuals_preds$actuals))
acc_err_table = cbind(min_max_accuracy, mape)
kable(acc_err_table, caption = "min max accuracy & mean absolute percentage error")

# a summary table on beta values of the multilevel MLR
model_multilevel_linear %>% broom.mixed::tidy() %>% select(c("term", "effect", "group", "estimate")) %>% kable()


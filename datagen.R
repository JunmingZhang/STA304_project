# author: Junming Zhang
# code used to generate the data to build the model from the raw data
# and generate plots of the data distribution with the basic RDD and
# multivariate inear model regression (lm)

library("tidyverse")

# read the raw data we need to produce the data sets we need
read_data <- function(work_path) {
  # set the working directory of the data
  setwd(work_path)
  
  # read the data for responses
  hiv_increase_data = read.csv("../data/new-cases-of-hiv-infection.csv")
  population_data = read.csv("../data/world-population-by-world-regions-post-1820.csv")
  
  # read the data for predictors
  hdi_data = read.csv("../data/human-development-index.csv")
  haq_data = read.csv("../data/healthcare-access-and-quality-index.csv")
  gdp_per_capita_data = read.csv("../data/average-real-gdp-per-capita-across-countries-and-regions.csv")
  female_emply_data = read.csv("../data/recent-ILO-LFP.csv")
  urbanization_data = read.csv("../data/share-of-population-urban.csv")
  drug_and_alcohol_data = read.csv("../data/share-with-alcohol-or-drug-use-disorders.csv")
  
  # return the loaded data to generate data sets we need
  out = list(hiv=hiv_increase_data, pop=population_data, hdi=hdi_data, haq=haq_data,
             gdp=gdp_per_capita_data, female=female_emply_data,
             urb=urbanization_data, disorder=drug_and_alcohol_data)
  return(out)
}

# combine all data frames into 1 large table and compute the new data necessary for building the model
data_generator <- function(work_path) {
  # load the data
  out = read_data(work_path)
  
  hiv_increase_data = out$hiv
  population_data = out$pop
  
  hdi_data = out$hdi
  haq_data = out$haq
  gdp_per_capita_data = out$gdp
  female_emply_data = out$female
  urbanization_data = out$urb
  drug_and_alcohol_data = out$disorder
  
  # the range of years used for training and testing the model, with step length 5
  # pred_years: the range of years for predictors ()
  pred_years = seq(from=1990, to=2010, by=5)
  # resp_years: the range of years for responses
  resp_years = seq(from=1995, to=2015, by=5)
  
  # filter out the rows we need to build the model (years in the range with the step length)
  hiv_increase_data %>% filter(Year %in% resp_years) %>%
    rename("new_cases_5_years_later" = colnames(hiv_increase_data)[length(colnames(hiv_increase_data))]) %>%
    select(c("Year", "Entity", "new_cases_5_years_later")) %>% mutate(Year = Year - 5) -> hiv_increase_data
  population_data %>% filter(Year %in% resp_years) %>%
    rename("population_size_5_years_later" = colnames(population_data)[length(colnames(population_data))]) %>%
    select(c("Year", "Entity", "population_size_5_years_later")) %>% mutate(Year = Year - 5) -> population_data
  
  hdi_data %>% filter(Year %in% pred_years) %>%
    rename("hdi" = colnames(hdi_data)[length(colnames(hdi_data))]) %>%
    select(c("Year", "Entity", "hdi")) -> hdi_data
  haq_data %>% filter(Year %in% pred_years) %>%
    rename("haq" = colnames(haq_data)[length(colnames(haq_data))]) %>%
    select(c("Year", "Entity", "haq")) -> haq_data
  gdp_per_capita_data %>% filter(Year %in% pred_years) %>%
    rename("gdp_per_capita" = colnames(gdp_per_capita_data)[length(colnames(gdp_per_capita_data))]) %>%
    select(c("Year", "Entity", "gdp_per_capita")) -> gdp_per_capita_data
  female_emply_data %>% filter(Year %in% pred_years) %>%
    rename("female_emply_rate" = colnames(female_emply_data)[length(colnames(female_emply_data))]) %>%
    select(c("Year", "Entity", "female_emply_rate")) -> female_emply_data
  urbanization_data %>% filter(Year %in% pred_years) %>%
    rename("urb_rate" = colnames(urbanization_data)[length(colnames(urbanization_data))]) %>%
    select(c("Year", "Entity", "urb_rate")) -> urbanization_data
  drug_and_alcohol_data %>% filter(Year %in% pred_years) %>%
    rename("drug_alcohol_disorder_share" = colnames(drug_and_alcohol_data)[length(colnames(drug_and_alcohol_data))]) %>%
    select(c("Year", "Entity", "drug_alcohol_disorder_share")) -> drug_and_alcohol_data

  # natural join all datasets, so we combine the dataset to a large one
  join_key = c("Year", "Entity")
  hiv_increase_data %>% inner_join(population_data, by=join_key) %>%
  inner_join(hdi_data, by=join_key) %>% inner_join(haq_data, by=join_key)  %>%
    inner_join(gdp_per_capita_data, by=join_key) %>% inner_join(female_emply_data, by=join_key) %>%
    inner_join(urbanization_data, by=join_key) %>% inner_join(drug_and_alcohol_data, by=join_key) -> large_table
  
  # compute the new data we need to build the model (I plan to use the RDD method to make causal inference with the observational data)
  large_table %>% mutate(new_cases_share_5_years_later=new_cases_5_years_later / population_size_5_years_later) %>%
    mutate(hdi_avg=mean(hdi)) %>% mutate(haq_avg=mean(haq)) %>% mutate(gdp_per_capita_avg=mean(gdp_per_capita)) %>%
    mutate(female_emply_rate_avg=mean(female_emply_rate)) %>% mutate(urb_rate_avg=mean(urb_rate)) %>% mutate(drug_alcohol_disorder_share_avg=mean(drug_alcohol_disorder_share)) -> large_table
  large_table %>% mutate(hdi_over_avg=ifelse(hdi > hdi_avg, 1, 0)) %>% mutate(haq_over_avg=ifelse(haq > haq_avg, 1, 0)) %>%
    mutate(gdp_per_capita_over_avg=ifelse(gdp_per_capita > gdp_per_capita_avg, 1, 0)) %>% mutate(female_emply_rate_over_avg=ifelse(female_emply_rate > female_emply_rate_avg, 1, 0)) %>%
    mutate(urb_rate_over_avg=ifelse(urb_rate > urb_rate_avg, 1, 0)) %>% mutate(drug_alcohol_disorder_share_over_avg=ifelse(drug_alcohol_disorder_share > drug_alcohol_disorder_share_avg, 1, 0)) -> large_table
  
  large_table$Entity = as.factor(large_table$Entity)
  return(large_table)
}

# split the datasets to the train and test data set
split_data <- function(work_path) {
  large_table = data_generator(work_path)
  large_table %>% filter(Year < 2010) -> train_set
  large_table %>% filter(Year == 2010) -> test_set
  return(list(train=train_set, test=test_set))
}

work_path = "/Users/peterzhang/Desktop/UTSG/STA/project/work"
setwd(work_path)

data_sets = split_data(work_path)
train_set = data_sets$train
test_set = data_sets$test

# export the dataset I generated to csvs
write.csv(train_set, paste("../data", "train_set.csv", sep="/"))
write.csv(test_set, paste("../data", "test_set.csv", sep="/"))

require(gridExtra)
# plots on the train data and test data
train_set %>% 
  ggplot(aes(x = hdi,
             y = new_cases_share_5_years_later * 1000)) +
  geom_point(alpha = 0.2) +
  geom_smooth(data = train_set %>% filter(hdi_over_avg==0), 
              method='lm',
              color = "black") +
  geom_smooth(data = train_set %>% filter(hdi_over_avg==1), 
              method='lm',
              color = "red") +
  theme_minimal() +
  ggtitle("HDI vs New infection cases share") +
  labs(x = "HDI",
       y = "New Cases Share * 1000") ->train_hdi

train_set %>% 
  ggplot(aes(x = haq,
             y = new_cases_share_5_years_later * 1000)) +
  geom_point(alpha = 0.2) +
  geom_smooth(data = train_set %>% filter(haq_over_avg==0), 
              method='lm',
              color = "black") +
  geom_smooth(data = train_set %>% filter(haq_over_avg==1), 
              method='lm',
              color = "red") +
  theme_minimal() +
  ggtitle("HAQ vs New infection cases share") +
  labs(x = "HAQ",
       y = "New Cases Share * 1000") ->train_haq

train_set %>% 
  ggplot(aes(x = gdp_per_capita,
             y = new_cases_share_5_years_later * 1000)) +
  geom_point(alpha = 0.2) +
  geom_smooth(data = train_set %>% filter(gdp_per_capita_over_avg==0), 
              method='lm',
              color = "black") +
  geom_smooth(data = train_set %>% filter(gdp_per_capita_over_avg==1), 
              method='lm',
              color = "red") +
  theme_minimal() +
  ggtitle("GDP per capita vs New infection cases share") +
  labs(x = "GDP per capita",
       y = "New Cases Share * 1000") ->train_gdp

train_set %>% 
  ggplot(aes(x = female_emply_rate,
             y = new_cases_share_5_years_later * 1000)) +
  geom_point(alpha = 0.2) +
  geom_smooth(data = train_set %>% filter(female_emply_rate_over_avg==0), 
              method='lm',
              color = "black") +
  geom_smooth(data = train_set %>% filter(female_emply_rate_over_avg==1), 
              method='lm',
              color = "red") +
  theme_minimal() +
  ggtitle("Female labour force participation vs New infection cases share") +
  labs(x = "female employment rate",
       y = "New Cases Share * 1000") ->train_female_emply

train_set %>% 
  ggplot(aes(x = urb_rate,
             y = new_cases_share_5_years_later * 1000)) +
  geom_point(alpha = 0.2) +
  geom_smooth(data = train_set %>% filter(urb_rate_over_avg==0), 
              method='lm',
              color = "black") +
  geom_smooth(data = train_set %>% filter(urb_rate_over_avg==1), 
              method='lm',
              color = "red") +
  theme_minimal() +
  ggtitle("Urbanization rate vs New infection cases share") +
  labs(x = "urbanization rate",
       y = "New Cases Share * 1000") ->train_urb

train_set %>%
  ggplot(aes(x = drug_alcohol_disorder_share,
             y = new_cases_share_5_years_later * 1000)) +
  geom_point(alpha = 0.2) +
  geom_smooth(data = train_set %>% filter(drug_alcohol_disorder_share_over_avg==0),
              method='lm',
              color = "black") +
  geom_smooth(data = train_set %>% filter(drug_alcohol_disorder_share_over_avg==1),
              method='lm',
              color = "red") +
  theme_minimal() +
  ggtitle("Drug & alcohol use disorder share vs New infection cases share") +
  labs(x = "Drug/alcohol use disorder share",
       y = "New Cases Share * 1000") -> train_disorder

grid.arrange(train_hdi, train_haq, train_gdp, train_female_emply, train_urb, train_disorder) -> data_dist_plots


# author: Junming Zhang
# build a multilevel linear regression model with the generated data
# with the train set and test against the test set

work_path = "/Users/peterzhang/Desktop/UTSG/STA/project/work"
setwd(work_path)

train_set = read.csv("../data/train_set.csv")
test_set = read.csv("../data/test_set.csv")

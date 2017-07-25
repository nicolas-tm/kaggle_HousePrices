# Start the clock!
start_time <- Sys.time()

library('caret')
library('data.table')
library('Metrics')

setwd("~/repo/kaggle/kaggle_HousePrices/src")
set.seed(2017)

# Read Data
Training <- fread("../input/train.csv", stringsAsFactors=TRUE)
Test     <- fread("../input/test.csv", stringsAsFactors=TRUE) 

intrain   <- createDataPartition(y=Training$SalePrice , p=0.7, list=FALSE)
trtrain   <- Training[intrain,] ; trtest    <- Training[-intrain,] 

fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)
glm_model  <- caret::train(SalePrice ~ LotArea + 
                                          OverallQual + 
                                          OverallCond + 
                                          YearBuilt , 
                           data=trtrain, 
                           method = "glmboost",
                           preProcess = c("center", "scale"),
                           trControl = fitControl,
                           metric="RMSE")
summary(glm_model)
p <- predict(glm_model, newdata= trtest)
postResample(pred = p, obs = trtest$SalePrice)


preds     <- predict(glm_model, newdata=Test)
sol       <- data.frame(Id=as.integer(Test$Id),SalePrice=preds)
write.csv(sol ,"../submission/glmSol.csv",row.names=F, quote=F)
print( difftime( Sys.time(), start_time, units = 'min'))





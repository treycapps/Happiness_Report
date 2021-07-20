#Install Packages
library(caret)
library(tidyverse)
library(ggplot2)
library(car)

#Load the data
train_df <- read_csv('train.csv')
test_df <- read_csv('test.csv')
train_df <- train_df %>% select (-c(Country, `Dystopia Residual`, `Happiness Rank`))
test_df <- test_df %>% select (-c(Country, `Dystopia Residual`, `Happiness Rank`))

#Create the full model
fit_full <- lm(`Happiness Score` ~ ., data = train_df)
summary(fit_full)
#Remove the linear dependent columns
train_df <- train_df %>% select(-c(Year_2019, `Region_Western Europe`))

#Rerun fit_full
fit_full <- lm(`Happiness Score` ~ ., data = train_df)

#Check multicollinearity
vif(fit_full)
#No values are over 10 so no concerns at the moment.

#Determine the best predictors
#Set the starting position for variable selection
no_model <- lm(`Happiness Score` ~ 1, data=train_df)
#Feature selection using forward selection
forward <- step(no_model, direction='forward', scope=formula(fit_full), trace=0)
forward$coefficients
#Feature selection using stepwise selection
stepwise <- step(no_model, direction='both', scope=formula(fit_full), trace=0)
stepwise$coefficients
#Both selection method give the same predictors. 

#Select variables for training
selected_vars <- train_df %>% select(-c(`Region_Eastern Asia`, `Region_Middle East and Northern Africa`, `Region_Southern Asia`, Year_2018))

#Check assumptions for linear regression
lm_final <- lm(`Happiness Score` ~ ., data=selected_vars)
plot(lm_final)
#Residuals have no pattern and have constant varaince. The QQ-plot suggest the data came from a normal distribution.
cooks.distance(lm_final)
#No values are significant, there are no outliers or influential points in the data.
durbinWatsonTest(lm_final)
#Coefficent close to 2 was found suggesting no autocorrelation.
vif(lm_final)
#Once again there does not appear to be multicollinearity between the predictors.

#Set up 5 fold cross validation to evaluate the performance of model and reduce overfitting to the train set.
set.seed(123)
control <- trainControl(method = "cv", number = 5)
lm_fit <- train(`Happiness Score` ~ ., data = selected_vars,
                method = "lm",
                trControl = control)
lm_fit$results

#Visualize the important predictors
important_vars <- varImp(lm_fit)
plot(important_vars)

#Consider SVR model, find the best cost function using gird search
#(Run the model with 1 to find the optimal value for cost function)
svr_fit <- train(`Happiness Score` ~ ., data = selected_vars,
                method = "svmLinear",
                #1.tuneGrid = expand.grid(C = seq(0, 2, length = 20)),
                #2.tuneGrid = data.frame(C = 0.2105263),
                trControl = control)
svr_fit$results
plot(svr_fit)
#Rerun the SVR with C = 0.2105263
#Results are almost identical to the MLR
#We can conduct a T-test to see if there is a significant difference
compare_models(lm_fit, svr_fit)
#Fail to reject the null, so there is insufficent evidence to reject the null which states there is no difference in the mean performance of the models.
#We will predict on the multiple linear regression because all the assumptions have been met. 

#Evaluating the final model
test_df <- test_df %>% select(-c(Year_2019, `Region_Western Europe`, `Region_Eastern Asia`, `Region_Middle East and Northern Africa`, `Region_Southern Asia`, Year_2018))
predict <- predict(lm_fit, newdata = test_df)
eval <- postResample(pred = predict, obs = test_df$`Happiness Score`)
eval
normalized_rmse <- eval[1] / (mean(test_df$`Happiness Score`))
normalized_rmse
#We know 79.37% of the variability in Happiness Score is explained by the linear model. 
#The RMSE is relatively low suggesting a pretty high accuracy when predicting on new data.
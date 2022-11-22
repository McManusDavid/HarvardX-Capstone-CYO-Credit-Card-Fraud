#Havardx PH125.9x Capstone Project: Choose Your Own Project: Creditcard Fraud Detection 

#Install Keras 
##Should be done at the very start to avoid issues
if(!require(keras))install.packages("keras", repos="https://cran.rstudio.com/")
library(keras)
install_keras(envname = "r-reticulate")

# Installing Required Packages
if(!require(jsonlite))install.packages("jsonlite", repos="https://cran.rstudio.com/")
if(!require(tidyverse))install.packages("tidyverse", repos = "https://cran.rstudio.com/")
if(!require(tidyr))install.packages('tidyr', repos = "https://cran.rstudio.com/")
if(!require(dplyr))install.packages('dplyr', repos = "https://cran.rstudio.com/")
if(!require(kableExtra))install.packages("kableExtra", repos = "https://cran.rstudio.com/")
if(!require(ggplot2))install.packages("ggplot2", repos = "https://cran.rstudio.com/")
if(!require(ROSE))install.packages("ROSE", repos = "https://cran.rstudio.com/")
if(!require(caret))install.packages('caret', repos = "https://cran.rstudio.com/")
if(!require(pROC))install.packages('pROC', repos = "https://cran.rstudio.com/")
if(!require(ranger))install.packages('ranger', repos = "https://cran.rstudio.com/")
if(!require(recipes))install.packages("recipes", repos = "https://cran.rstudio.com/")
if(!require(scales))install.packages("scales", repos = "https://cran.rstudio.com/")



# tensorflow/kears install


library("jsonlite")
library("tidyverse")
library("tidyr")
library("dplyr")
library("kableExtra")
library("ggplot2")
library("ROSE")
library("caret")
library("pROC")
library("ranger")
library("recipes")
library("scales")
library("keras")
library("tensorflow")

#set seed
set.seed(1, sample.kind = "Rounding")
set_random_seed(1) #sets seed for keras/tensorflow

# Change memory allocation 
# 56000 equals 7gb
memory.limit(size=56000)

##########################################################################

#Creating data set 
# code from https://datahub.io/machine-learning/creditcard#r
json_file <- 'https://datahub.io/machine-learning/creditcard/datapackage.json'
json_data <- fromJSON(paste(readLines(json_file), collapse=""))

# get list of all resources:
print(json_data$resources$name)

# print all tabular data(if exists any)
for(i in 1:length(json_data$resources$datahub$type)){
  if(json_data$resources$datahub$type[i]=='derived/csv'){
    path_to_file = json_data$resources$path[i]
    data <- read.csv(url(path_to_file))
    print(data)
  }
}

#tidy workspace
rm(json_data, json_file, path_to_file)

#####################################################
##Analysis
#check sturcture of dataset

##scrollbox for html
##use kable_styling(latex_options = c("striped", "hold_position", "scale_down")) for pdf
head(data)%>%
  kbl(caption = "head of Dataset") %>%
  kable_styling(latex_options = c("striped"))%>%
  scroll_box(height = "100%", width = "100%", fixed_thead = TRUE)


summary(data)%>%
  kbl(caption = "Summary of Dataset") %>%
  kable_styling(latex_options = c("striped"))%>%
  scroll_box(height = "100%", width = "100%", fixed_thead = TRUE)

##check for na's
anyNA(data) %>%
  kbl(col.names = "Any NA's") %>%
  kable_styling(latex_options = c("striped"))

#amount of fraud/legitimate
table(data$Class) %>% 
  kbl(caption = "Number of Real and Fraudulent Transactions") %>%
  kable_styling(latex_options = c("striped"))

##proportions of f/l
prop.table(table(data$Class))%>%
  kbl(caption = "Frequency of Real and Fraudulent Transactions") %>%
  kable_styling(latex_options = c("striped"))
  
##barchart amount of legit v fraud
data$Class %>% 
  as.factor() %>% 
  as.data.frame() %>%
  ggplot(aes(., fill = .)) +
  geom_bar(show.legend = FALSE)+
  scale_y_continuous(labels = comma) +
  ggtitle("Distribution of Class")+
  scale_x_discrete(labels = c("Legitimate", "Fraud"))

#checking for correlations with class
#amount
#amount v Class
#V4, V9, V11, V15 V Class
#V4
data %>% 
  ggplot(aes(V4, fill = Class)) +
  geom_density(alpha = 0.2) +
  scale_x_log10()+
  ggtitle("V4 by Class Scaled")+
  scale_fill_discrete(labels=c('Legitimate', 'Fraud'))

#V9
data %>% 
  ggplot(aes(V9, fill = Class)) +
  geom_density(alpha = 0.2) +
  scale_x_log10()+
  ggtitle("V9 by Class Scaled")+
  scale_fill_discrete(labels=c('Legitimate', 'Fraud'))

#V11
data %>% 
  ggplot(aes(V11, fill = Class)) +
  geom_density(alpha = 0.2) +
  scale_x_log10()+
  ggtitle("V11 by Class Scaled")+
  scale_fill_discrete(labels=c('Legitimate', 'Fraud'))


#V15
data %>% 
  ggplot(aes(V15, fill = Class)) +
  geom_density(alpha = 0.2) +
  scale_x_log10()+
  ggtitle("V15 by Class Scaled")+
  scale_fill_discrete(labels=c('Legitimate', 'Fraud'))



# Creating Validation, Train and Test sets
# do val at 20%, then test at 20%, then use rose to balance

#Validation set
test_index <- createDataPartition(y = data$Class, times = 1, p = 0.2, list = FALSE)
imbal_set <- data[-test_index, ]
validation <- data[test_index, ]

#train and test sets

test_index <- createDataPartition(y = imbal_set$Class, times = 1, p = 0.2, list = FALSE)
imbal_train <- imbal_set[-test_index, ]
test_set <- imbal_set[test_index, ]

rm(imbal_set, test_index)
#Need to address the extreme imbalance in dataset
#use ROSE package

#over sampling with ROSE
data.rose <- ROSE(Class ~ ., data = imbal_train, seed = 1)$data

#checking new dataset
table(data.rose$Class) %>% 
  kbl(caption = "Number of Real and Fraudulent Transactions in Balanced Dataset") %>%
  kable_styling(latex_options = c("striped"), full_width = F)

prop.table(table(data.rose$Class))%>%
  kbl(caption = "Frequency of Real and Fraudulent Transactions in Balanced Dataset") %>%
  kable_styling(latex_options = c("striped"), full_width = F)

rm(imbal_train)

###relevel class and remove time
#renaming levels and relevel, remove time
#data rose
data.rose$Class <- as.factor(data.rose$Class)
levels(data.rose$Class)[levels(data.rose$Class)=="'1'"]<- 'Fraud'
levels(data.rose$Class)[levels(data.rose$Class)=="'0'"]<- 'Good'

data.rose$Class <- relevel(data.rose$Class, "Fraud") 
data.rose <- data.rose %>% select(-Time)

#test set
test_set$Class <- as.factor(test_set$Class)
levels(test_set$Class)[levels(test_set$Class)=="'1'"]<- 'Fraud'
levels(test_set$Class)[levels(test_set$Class)=="'0'"]<- 'Good'

test_set$Class <- relevel(test_set$Class, "Fraud") 
test_set <- test_set %>% select(-Time)

#val set
validation$Class <- as.factor(validation$Class)
levels(validation$Class)[levels(validation$Class)=="'1'"]<- 'Fraud'
levels(validation$Class)[levels(validation$Class)=="'0'"]<- 'Good'

validation$Class <- relevel(validation$Class, "Fraud") 
validation <- validation %>% select(-Time)

############################################################################

#detection sysytems 
## Set traincontrol
trainControl <- trainControl(method = "cv",
                             number = 10,
                             classProbs = TRUE,
                             summaryFunction = twoClassSummary
)
                             

##GLM
##<2min
glm_fit <- caret::train(
  Class ~ .,
  data = data.rose,
  trControl = trainControl,
  method = "glm",
  family = "binomial",
  metric = "ROC"
)

predict_test <- data.frame(
  obs = test_set$Class,
  predict(glm_fit, newdata = test_set, type = "prob"),
  pred = predict(glm_fit, newdata = test_set, type = "raw")
)
predict_test$pred <- relevel(predict_test$pred, "Fraud")
glm_auc <- twoClassSummary(data = predict_test, lev = levels(predict_test$obs))

#record and print results
glm_cf <- confusionMatrix(predict_test$pred, test_set$Class)
glm_cf
auc_results <- tibble(Method = "glm", AUC  = glm_auc[1], Sensitivity = glm_auc[2], Specificity = glm_auc[3])

auc_results %>% 
  kbl(label = "Area Under Curve Results") %>%
  kable_styling(latex_options = c("striped"), full_width = F)


#Decision Tree - rpart
#10=15 min
rpart_fit <- caret::train(Class ~ ., 
                          data = data.rose,
                          method = "rpart",
                          trControl = trainControl,
                          metric = "ROC"
)


predict_test <- data.frame(
  obs = test_set$Class,
  predict(rpart_fit, newdata = test_set, type = "prob"),
  pred = predict(rpart_fit, newdata = test_set, type = "raw")
)
predict_test$pred <- relevel(predict_test$pred, "Fraud")
rpart_auc <- twoClassSummary(data = predict_test, lev = levels(predict_test$obs))

##record and print results
rpart_cf <- confusionMatrix(predict_test$pred, test_set$Class)
rpart_cf

auc_results <- bind_rows(auc_results, tibble(Method = "Decision Tree - rpart", AUC  = rpart_auc[1], Sensitivity = rpart_auc[2], Specificity = rpart_auc[3]))

auc_results %>% 
  kbl(label = "Area Under Curve Results") %>%
  kable_styling(latex_options = c("striped"), full_width = F)

##Random Forest - ranger
##10 fold takes too long, use 2 instead 
#new traincontrol
trainControl2 <- caret::trainControl(method = "cv",
                                     number = 2,
                                     classProbs = TRUE,
                                     summaryFunction = twoClassSummary,
                                     verbose = FALSE
)

tuneGrid <- data.frame(
  splitrule = "gini",
  min.node.size = 1
)

ranger_fit <- caret::train(Class ~ .,
                           data = data.rose,
                           method = "ranger",
                           trControl = trainControl2,
                           metric = "ROC",
                           verbose = FALSE)


predict_test <- data.frame(
  obs = test_set$Class,
  predict(ranger_fit, newdata = test_set, type = "prob"),
  pred = predict(ranger_fit, newdata = test_set, type = "raw")
)

predict_test$pred <- relevel(predict_test$pred, "Fraud")
ranger_auc <- twoClassSummary(data = predict_test, lev = levels(predict_test$obs))

ranger_cf <- confusionMatrix(predict_test$pred, test_set$Class)

##record and print results
ranger_cf 
auc_results <- bind_rows(auc_results, tibble(Method = "Random Forest - Ranger", AUC  = ranger_auc[1], Sensitivity = ranger_auc[2], Specificity = ranger_auc[3]))

auc_results %>% 
  kbl(label = "Area Under Curve Results") %>%
  kable_styling(latex_options = c("striped"), full_width = F)


### knn3
## using caret takes way too long (>hour)
## use as stand alone - no difference in auc
knn3_cv <- knn3(Class ~ ., data.rose)

predict_test <- data.frame(
  obs = test_set$Class,
  predict(knn3_cv, newdata = test_set, type = "prob"),
  pred = predict(knn3_cv, newdata = test_set, type = "class")
)

predict_test$pred <- relevel(predict_test$pred, "Fraud")
knn_auc <- twoClassSummary(data = predict_test, lev = levels(predict_test$obs))

auc_results <- bind_rows(auc_results, tibble(Method = "KNN3", AUC  = knn_auc[1], Sensitivity = knn_auc[2], Specificity = knn_auc[3]))
knn_cf <- confusionMatrix(predict_test$pred, test_set$Class)

##receord and print results
knn_cf
auc_results %>% 
  kbl(label = "Area Under Curve Results") %>%
  kable_styling(latex_options = c("striped"), full_width = F)

################################################################################
##keras sequential model
#using recipes to center and scale
rec_obj <- recipe(Class ~., data = data.rose)%>%
  step_center(all_predictors(), -all_outcomes())%>%
  step_scale(all_predictors(), - all_outcomes())%>%
  prep(data = data.rose)


#baking the data.rose and test set, and seperating class
x_train <- bake(rec_obj, new_data = data.rose) %>% select(-Class)
x_test <- bake(rec_obj, new_data = test_set) %>% select(-Class)

##keras needs target to be seperate and numeric 
y_train <- ifelse(pull(data.rose, Class)== "Fraud", 1, 0)
y_test <- ifelse(pull(test_set, Class)== "Fraud", 1, 0)

seq_mod <- keras_model_sequential()
#add layers
seq_mod %>%
  #first layer
  layer_dense(
    units = 80,
    activation = "relu",
    input_shape = ncol(x_train)
  ) %>%
  #add dropout to prevent overfitting
  layer_dropout(rate = 0.1)%>%
  #second layer
  layer_dense(
    units = 75,
    activation = "relu"
  )%>%
  layer_dropout(rate = 0.1)%>%
  #third layer
  layer_dense(
    units = 35,
    activation = "relu"
  )%>%
  layer_dropout(rate = 0.1)%>%
  #fourth layer
  
  layer_dense(
    units = 16,
    activation = "relu"
  ) %>%
  
  layer_dropout(rate = 0.1) %>%
  layer_dense(
    units = 12,
    activation = "relu"
  )%>%
  
  layer_dropout(rate = 0.1) %>%
  layer_dense(
    units = 4,
    activation = "relu"
  )%>%
  #output layer
  layer_dense(
    units = 1,
    activation = "sigmoid" # 'softmax', 'elu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', linear'.
  )%>%
  #compile
  compile(
    optimizer = optimizer_adamax(), 
    #adam , SGD , RMSprop , adadelta , adagrad  , adamax  , nadam errors , ftrl errors
    loss = "binary_crossentropy", #with adamax - binary_crossentropy  , categorical_crossentropy error,
    metrics = list(metric_auc(), 
                   metric_false_negatives(), 
                   metric_false_positives(), 
                   metric_true_negatives(), 
                   metric_true_positives()) #'accuracy', 'binary_accuracy
  )

#https://www.rdocumentation.org/packages/kerasR/versions/0.8.1/topics/keras_compile

##create cp path to save weights
checkpoint_path <- "training_2/cp-list{epoch:04d}.ckpt"
checkpoint_dir <- fs::path_dir(checkpoint_path)
batch_size <- 50
##cp callback
cp_callback <- callback_model_checkpoint(
  filepath = checkpoint_path,
  verbose = 1,
  save_weights_only = TRUE,
  save_freq = 5*batch_size
)

##fit the model
history <- fit(
  object = seq_mod,
  x = as.matrix(x_train),
  y = y_train,
  batch_size = 50,
  epochs = 10,
  validation_split = 0.20,
  callbacks = cp_callback
)

## reload the weights
latest <- tf$train$latest_checkpoint(checkpoint_dir)

load_model_weights_tf(seq_mod, latest)

## make predictions
keras_predict <- seq_mod %>%predict(as.matrix(x_test)) %>% '>'(0.5) %>% k_cast("int32")
keras_table <-table(as.numeric(keras_predict[,1]), as.numeric(y_test))

keras_auc <- auc(as.numeric(keras_predict[,1]), as.numeric(y_test))
#keras_auc #0.9995
keras_table <-table(as.numeric(keras_predict[,1]), as.numeric(y_test))

keras_cf <- confusionMatrix(keras_table, positive = "1")

##record and print results
keras_cf

auc_results <- bind_rows(auc_results, tibble(Method = "keras Sequential Model", AUC  = keras_auc[1], Sensitivity = keras_cf$byClass[1], Specificity = keras_cf$byClass[2]))

auc_results %>% 
  kbl(label = "Area Under Curve Results") %>%
  kable_styling(latex_options = c("striped"), full_width = F)

##########################################################################
#table with keras/glm showing fp/fn's
keras_v_glm <- tibble(Method = "keras Sequential Model", 
                      AUC  = keras_auc[1], 
                      Sensitivity = keras_cf$byClass[1], 
                      Specificity = keras_cf$byClass[2],
                      FP = keras_cf$table[2,1],
                      FN = keras_cf$table[1,2])

keras_v_glm <- bind_rows(keras_v_glm, 
                         tibble(Method = "glm Model", 
                                AUC  = glm_auc[1], 
                                Sensitivity = glm_cf$byClass[1], 
                                Specificity = glm_cf$byClass[2],
                                FP = glm_cf$table[1,2],
                                FN = glm_cf$table[2,1]))

keras_v_glm %>% 
  kbl(label = "keras v glm") %>%
  kable_styling(latex_options = c("striped"), full_width = F)%>%
  footnote(general = " FP = False Positivies, FN = False Negatives")

########################################################################
##Final Validation
#bake val set and seperate target
x_val <- bake(rec_obj, new_data = validation) %>% select(-Class)

y_val <- ifelse(pull(validation, Class)== "Fraud", 1, 0)

##reload weights
latest <- tf$train$latest_checkpoint(checkpoint_dir)

load_model_weights_tf(seq_mod, latest)

## make predictions
val_predict <- seq_mod %>% predict(as.matrix(x_val)) %>% '>'(0.5) %>% k_cast("int32")
val_auc <- auc(as.numeric(val_predict), as.numeric(y_val))

val_table <-table(as.numeric(val_predict), as.numeric(y_val))
val_cf <- confusionMatrix(val_table, positive = "1")

##record and print results
val_cf 

auc_results <- bind_rows(auc_results, tibble(Method = "Final Validation - Keras Sequential", AUC  = val_auc[1], Sensitivity = val_cf$byClass[1], Specificity = val_cf$byClass[2]))

auc_results %>% 
  kbl(label = "Area Under Curve Results") %>%
  kable_styling(latex_options = c("striped"), full_width = F)

##################################################################################
#table with all models showing fp/fn's
results_FP_FN <- tibble(Method = "keras Sequential Model", 
                        AUC  = keras_auc[1], 
                        Sensitivity = keras_cf$byClass[1], 
                        Specificity = keras_cf$byClass[2],
                        FP = keras_cf$table[2,1],
                        FN = keras_cf$table[1,2])

results_FP_FN <- bind_rows(results_FP_FN, 
                           tibble(Method = "glm Model", 
                                  AUC  = glm_auc[1], 
                                  Sensitivity = glm_cf$byClass[1], 
                                  Specificity = glm_cf$byClass[2],
                                  FP = glm_cf$table[1,2],
                                  FN = glm_cf$table[2,1]))

results_FP_FN <- bind_rows(results_FP_FN, 
                           tibble(Method = "ranger Model", 
                                  AUC  = ranger_auc[1], 
                                  Sensitivity = ranger_cf$byClass[1], 
                                  Specificity = ranger_cf$byClass[2],
                                  FP = ranger_cf$table[1,2],
                                  FN = ranger_cf$table[2,1]))

results_FP_FN <- bind_rows(results_FP_FN, 
                           tibble(Method = "keras Sequential Model - Final Validation", 
                                  AUC  = val_auc[1], 
                                  Sensitivity = val_cf$byClass[1], 
                                  Specificity = val_cf$byClass[2],
                                  FP = val_cf$table[2,1],
                                  FN = val_cf$table[1,2]))


results_FP_FN <- bind_rows(results_FP_FN,  
                           tibble(Method = "rpart Model", 
                                  AUC  = rpart_auc[1], 
                                  Sensitivity = rpart_cf$byClass[1], 
                                  Specificity = rpart_cf$byClass[2],
                                  FP = rpart_cf$table[1,2],
                                  FN = rpart_cf$table[2,1]))


results_FP_FN <- bind_rows(results_FP_FN, 
                           tibble(Method = "knn3 Model", 
                                  AUC  = knn_auc[1], 
                                  Sensitivity = knn_cf$byClass[1], 
                                  Specificity = knn_cf$byClass[2],
                                  FP = knn_cf$table[1,2],
                                  FN = knn_cf$table[2,1]))

results_FP_FN %>% 
  kbl(label = "Results including False Positives and Negatives") %>%
  kable_styling(latex_options = c("striped"), full_width = F)%>%
  footnote(general = " FP = False Positivies, FN = False Negatives")

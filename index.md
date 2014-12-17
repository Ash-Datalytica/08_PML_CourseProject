# Course Project
**Practical Machine Learning**  
*by Brian Caffo, PhD, Jeff Leek, PhD, Roger D. Peng, PhD*  
*Johns Hopkins Bloomberg School of PULIC HEALTH*  

### Problem definition

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

In this project, the goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

### Getting data
For the first run we download CSV files with training and testing from the links and store them locally in the `/data/` subfolder of the working directory.

```r
trainURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
trainFile <- "./data/pml-training.csv"
testFile <- "./data/pml-testing.csv"
if (!file.exists(trainFile)) {
    download.file (trainURL, trainFile, mode="wb"); 
    trainDownloadDate <- Sys.Date()
}
if (!file.exists(testFile)) {
    download.file (testURL, testFile, mode="wb"); 
    testDownloadDate <- Sys.Date()
}
```

Use `fread()` for fast file reading. We consider value `"#DIV/0!"` to be equal to `NA`.

```r
library(data.table)
train <- fread(trainFile, sep = ",", header = TRUE, na.strings = c("", "NA", "#DIV/0!"), 
               stringsAsFactors = FALSE, data.table = FALSE)
test <- fread(testFile, sep = ",", header = TRUE, na.strings = c("", "NA", "#DIV/0!"), 
               stringsAsFactors = FALSE, data.table = FALSE)
```

### Cleaning data
We need to process loaded data for further analysis: give names to varables with no names, convert strings to factors, and convert numeric variables wich were recognised by `fread()` as character to be really numeric.


```r
names(train)[1] <- "id"
train$classe <- as.factor(train$classe)
train$new_window <- as.factor(train$new_window)
train$user_name <- as.factor(train$user_name)

notUsedColumns <- c(1,5) #id, cvtd_timestamp (date) - won't use as predictors and won't convert to num
charColumns <- which(sapply(train, class) == "character")
charColumns <- setdiff(charColumns, notUsedColumns )
train[,charColumns] <- apply(train[, charColumns], 2, as.numeric)

names(test)[1] <- "id"
test$new_window <- as.factor(test$new_window)
test$user_name <- as.factor(test$user_name)
test[,charColumns] <- apply(test[, charColumns], 2, as.numeric)
```

### Exploring data

First, we split initial train data into training and validating datasets. We use only 10% of train data for training and the remaining 90% go to valitating because it is enough to achive satisfactory accuracy and takes considerably smaller amount of time and computer memory to build the model then recommended 70/30 split.

Then, we investigate training subset.

```r
#Split data into training and validating subsets
library (caret)
set.seed (1234)
inTrain <- createDataPartition(y = train$classe, p=0.10, list=FALSE)
training <- train [inTrain,]
validating <- train [-inTrain,]

#size of data sets
dim (training); dim (validating)
```

```
## [1] 1964  160
```

```
## [1] 17658   160
```

```r
#outcome variable
summary(training$classe)
```

```
##   A   B   C   D   E 
## 558 380 343 322 361
```

```r
#non numeric variables
table(sapply(training, class))
```

```
## 
## character    factor   integer   numeric 
##         2         3        34       121
```

```r
names(training)[!(sapply(training, class) %in% c("numeric", "integer"))]
```

```
## [1] "id"             "user_name"      "cvtd_timestamp" "new_window"    
## [5] "classe"
```

### Data preprocessing
First, define  features (variables) wich don't hold enough variance. For example, if all values of a feature are NA, such feature is useless as a predictor. Moreover, algorithms which we are going to use (KNN, RF) dont work with such features. We won't use them further as predictors. We set very low cutoff level of 0.05% to leave more varibles as potential predictors.

```r
#Exclude near zero variance variables, e.g. all values are NA's
notUsedColumns <- union(notUsedColumns, 
                        nearZeroVar (training, saveMetrics=FALSE, uniqueCut = 0.05))
names(training)[notUsedColumns]
```

```
##  [1] "id"                     "cvtd_timestamp"        
##  [3] "kurtosis_yaw_belt"      "skewness_yaw_belt"     
##  [5] "amplitude_yaw_belt"     "kurtosis_yaw_dumbbell" 
##  [7] "skewness_yaw_dumbbell"  "amplitude_yaw_dumbbell"
##  [9] "kurtosis_yaw_forearm"   "skewness_yaw_forearm"  
## [11] "amplitude_yaw_forearm"
```

Second, we impute NA values for all numeric variables which have non zero variance. Do it with both training and validating datasets.

```r
#impute NA's for integer and numeric (not Factor or character)
numColumns <- setdiff(which(sapply(training, class) %in% c("numeric", "integer")), notUsedColumns)
preObj <- preProcess(training[,numColumns], method="knnImpute") #K-nearest neighbours imputation
training[,numColumns] <- predict(preObj, training[,numColumns])

## Process validating features
validating[,numColumns] <- predict(preObj, validating[,numColumns])
```

Third, we want to reduce training set by reducing the number of predictors. We could do this using principal component analysis method. PCA method doesn't work with Factor variables, so we run it on all numeric variables and then add to the dataset factors which we belive to be usefull - `user_name` and `new_window`.

```r
preProc <- preProcess(training[,numColumns], method="pca", thresh=0.99)
preProc 
trainingPC <- predict(preProc, training[,numColumns])
#return Factor and outcome columns into the training set
trainingPC <- cbind(trainingPC,
                     training[,c("user_name", "new_window", "classe")])

#get PC for validating dataset based on the training preprocessing
validatingPC <- predict (preProc, validating[,numColumns]) 
#Put back Factor columns into the Validating set
validatingPC <- cbind(validatingPC,
                     validating[,c("user_name", "new_window", "classe")])
```

Experiments with models built on the reduced by PCA dataset have shown that appropriate performance of Accuracy = 96% could be achived when we train the model on the 75% of all data with PCA threshold equal to 99.9% (106 predictors). In this case training process **takes a lot of time and computer memory**.  
We choose to use repeated k-fold cross validation as one of the best methods according to [this](http://appliedpredictivemodeling.com/blog/2014/11/27/vpuig01pqbklmi72b8lcl3ij5hj2qm) Max Kuhn's article. 

```r
library("doSNOW")
# Assign number of cores you want to use; in this case use 5 cores. 
cl<-makeCluster(5) 
registerDoSNOW(cl) # Register the cores.

nPredictors <- dim (trainingPC)[2]-1
#Using repeated CV, http://appliedpredictivemodeling.com/blog/2014/11/27/vpuig01pqbklmi72b8lcl3ij5hj2qm
system.time ({
    modFit <- train (classe ~ . , 
                 data=trainingPC, method = "rf", prox=TRUE,
                 trControl = trainControl (method="cv", number = 10, repeats = 10),
                 tuneGrid=data.frame(mtry=c(ceiling(sqrt(nPredictors)),
                                            ceiling(nPredictors/3),
                                            ceiling(nPredictors/2),
                                            ceiling(nPredictors*2/3),
                                            nPredictors))
                  )
}) 
modFit

#check Accuracy
confusionMatrix(predict (modFit, newdata = validatingPC), validatingPC$classe) 
# Results:
#pca thresh=0.999, 106 predictors, 75%, 2 threads - 2.8 часа, mtry=11 (sqrt), асс=.96
#pca thresh=0.999, 106 predictors, 10%, 4 threads - 188 sec, mtry=25 (1/3), acc=.86
#pca thresh=0.99, 59 predictors, 10%, 4 threads - 160 sec, mtry=8 (sqrt), acc=.83
#pca thresh=0.99, 59 predictors, 10%, 6 threads - 137 sec, mtry=8 (sqrt), acc=.83
#pca thresh=0.99, 59 predictors, 10%, 6 threads - 140 sec, mtry=8 (sqrt), CVrepeats=30, acc=.83

stopCluster(cl) # Explicitly free up cores again.
```

Now let's try to build a model on the full predictors set, without usage of PCA reduction.

```r
library("doSNOW")
cl<-makeCluster(5) # Assign number of cores to use
registerDoSNOW(cl) # Register the cores.

#Using repeated CV, http://appliedpredictivemodeling.com/blog/2014/11/27/vpuig01pqbklmi72b8lcl3ij5hj2qm
system.time ({
    nPredictors <- dim (training[,-notUsedColumns])[2]-1
    modFit <- train (classe ~ . , 
                 data=training[,-notUsedColumns], method = "rf", prox=TRUE,
                 trControl = trainControl (method="cv", number = 10, repeats = 10),
                 tuneGrid=data.frame(mtry=c(
                     #ceiling(sqrt(nPredictors)), ceiling(nPredictors/3),
                    ceiling(nPredictors/2),
                    ceiling(nPredictors*2/3),
                    nPredictors
                    ))
                 )
}) 
```

```
##    user  system elapsed 
##   22.23    0.06  190.34
```

```r
modFit
```

```
## Random Forest 
## 
## 1964 samples
##  148 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 1768, 1767, 1768, 1768, 1766, 1767, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD  
##    74   0.9714800  0.9638946  0.01208423   0.01530227
##    99   0.9719773  0.9645200  0.01348208   0.01708127
##   148   0.9725005  0.9651834  0.01026973   0.01302332
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 148.
```

```r
#check Accuracy
cm <- confusionMatrix(predict (modFit, newdata = validating), validating$classe) 
cm 
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 5012   87    0    2    0
##          B    7 3272   62    1    1
##          C    0   50 3003   37   12
##          D    2    1   14 2846   23
##          E    1    7    0    8 3210
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9822          
##                  95% CI : (0.9801, 0.9841)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9774          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9980   0.9576   0.9753   0.9834   0.9889
## Specificity            0.9930   0.9950   0.9932   0.9973   0.9989
## Pos Pred Value         0.9826   0.9788   0.9681   0.9861   0.9950
## Neg Pred Value         0.9992   0.9899   0.9948   0.9968   0.9975
## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2838   0.1853   0.1701   0.1612   0.1818
## Detection Prevalence   0.2889   0.1893   0.1757   0.1634   0.1827
## Balanced Accuracy      0.9955   0.9763   0.9843   0.9904   0.9939
```

```r
#no_pca , 148 predictors, 5%, 6 threads - 83 sec, mtry=148 (100%), acc=.94
#no_pca , 148 predictors, 10%, 5 threads - 257 sec, mtry=148 (100%), acc=.98
#no_pca , 148 predictors, 20%, 5 threads - 748 sec, mtry=74 (1/2), acc=.99

stopCluster(cl) # Explicitly free up cores again.
```

This approach gives us similar results in terms of Accuracy (**98.22%**) but takes a lot less time to build the model. We chose this approch to be the final. So, out of sample error of the model is **1.78%**.

### Predicting classes
Now we'll use the model to predict values of `classe` variable for test dataset. 

```r
## Impute NA's in test features
test[,numColumns] <- predict(preObj, test[,numColumns])

## Predict answers
answers <- predict (modFit, newdata = test)
data.frame(problem_id = test$problem_id, classe = answers)
```

```
##    problem_id classe
## 1           1      B
## 2           2      A
## 3           3      B
## 4           4      A
## 5           5      A
## 6           6      E
## 7           7      D
## 8           8      B
## 9           9      A
## 10         10      A
## 11         11      B
## 12         12      C
## 13         13      B
## 14         14      A
## 15         15      E
## 16         16      E
## 17         17      A
## 18         18      B
## 19         19      B
## 20         20      B
```

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("submission/problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(answers)
```

All predicted values have past the submission test and are correct.

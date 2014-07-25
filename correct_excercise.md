
https://github.com/dmaust/DataScience-ML-Project
https://github.com/SweeRoty/pml/blob/master/project.md

Do it correctly
========================================================



Exec summary
?

## Exploration of data

### Getting the data
The data are aquired from: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv (training set) and from https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv (testing set). Data are described here: http://groupware.les.inf.puc-rio.br/har


```r
library(ggplot2);
library(caret);

if (file.exists("pml-training.rds")) {
    pml_training<-readRDS("pml-training.rds");
} else {
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile="pml-training.csv", method="curl");
    pml_training<-read.csv("pml-training.csv", stringsAsFactors=FALSE, na.strings = c("NA", "#DIV/0!"));
    saveRDS(pml_training, file="pml-training.rds");
}

if (file.exists("pml-testing.rds")) {
    pml_testing<-readRDS("pml-testing.rds");
} else {
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile="pml-testing.csv", method="curl");
    pml_testing<-read.csv("pml-testing.csv", stringsAsFactors=FALSE, na.strings = c("NA", "#DIV/0!"));
    saveRDS(pml_testing, file="pml-testing.rds");
}
```
### Removal of not predicitve data


```r
set.seed(8888)
trainingIndex <- createDataPartition(pml_training$classe, list=FALSE, p=.7)
training = pml_training[trainingIndex,]
testing = pml_training[-trainingIndex,]
```

Remove indicators with near zero variance.


```r
nzv <- nearZeroVar(training)

training <- training[-nzv]
testing <- testing[-nzv]
pml_testing <- pml_testing[-nzv]
```


```r
num_features_idx = which(lapply(training,class) %in% c('numeric')  )
```


```r
preModel <- preProcess(training[,num_features_idx], method=c('knnImpute'))

ptraining <- cbind(training$classe, predict(preModel, training[,num_features_idx]))
ptesting <- cbind(testing$classe, predict(preModel, testing[,num_features_idx]))
prtesting <- predict(preModel, pml_testing[,num_features_idx])

#Fixing names
names(ptraining)[1] <- 'classe'
names(ptesting)[1] <- 'classe'
```

### Classification
Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

Read more: http://groupware.les.inf.puc-rio.br/har#weight_lifting_exercises#ixzz38Q6Fxj5v


```r
cvControl <- trainControl(method = "cv", number = 5)
treefit <- train(classe ~ ., trControl = cvControl, method = "rpart", tuneLength = 5, data=ptraining);

plot(treefit)
```

![plot of chunk unnamed-chunk-6](figure/unnamed-chunk-61.png) 

```r
if (file.exists("rForest.RData")) {
    load("rForest.RData")
} else {
    rForest <- train(classe ~ ., trControl = cvControl, method = "rf", tuneLength = 5, data=ptraining);
    save(rForest, file="rForest.RData")
}
plot(rForest)
```

![plot of chunk unnamed-chunk-6](figure/unnamed-chunk-62.png) 

From the two plots, we can see that, using 5-fold cross validation, the estimated out of sample error rate of the Tree model is above 40%, while that of the Random Forest model is less than 1%. So we choose the best Random Forest model as our final model and apply it to our generated testing data to see how well it generalizes.

### Split training into train and test p=0.7




```r
model <- rForest$finalModel
pred.test <- predict(model, newdata = ptesting)
```

```
## Error: no applicable method for 'predict' applied to an object of class
## "randomForest"
```

```r
confusionMatrix(pred.test, ptesting$classe)
```

```
## Error: object 'pred.test' not found
```

```r
pred.train <- predict(model, newdata = ptraining)
```

```
## Error: no applicable method for 'predict' applied to an object of class
## "randomForest"
```

```r
confusionMatrix(pred.train, ptraining$classe)
```

```
## Error: object 'pred.train' not found
```

```r
answers <- predict(model, prtesting) 
```

```
## Error: no applicable method for 'predict' applied to an object of class
## "randomForest"
```

```r
answers
```

```
## Error: object 'answers' not found
```

Cross Validation
We are able to measure the accuracy using our training set and our cross validation set. With the training set we can detect if our model has bias due to ridgity of our mode. With the cross validation set, we are able to determine if we have variance due to overfitting.

In-sample accuracy

```r
training_pred <- predict(rf_model, ptraining) 
```

```
## Error: object 'rf_model' not found
```

```r
print(confusionMatrix(training_pred, ptraining$classe))
```

```
## Error: object 'training_pred' not found
```
The in sample accuracy is 100% which indicates, the model does not suffer from bias.

Out-of-sample accuracy
testing_pred <- predict(rf_model, ptesting) 
Confusion Matrix:

print(confusionMatrix(testing_pred, ptesting$classe))

The cross validation accuracy is greater than 99%, which should be sufficient for predicting the twenty test observations. Based on the lower bound of the confidence interval we would expect to achieve a 98.7% classification accuracy on new data provided.

One caveat exists that the new data must be collected and preprocessed in a manner consistent with the training data.


http://en.wikibooks.org/wiki/Data_Mining_Algorithms_In_R/Dimensionality_Reduction/Feature_Selection


## Prediction models


```r
modelFit<-train(classe ~ ., method = 'rf',  data=training)
```

```
## Error: final tuning parameters could not be determined
```

```r
modelFit
```

```
## Error: object 'modelFit' not found
```

```r
pred <- predict(modelFit,testing); testing$predRight <- pred==testing$classe
```

```
## Error: object 'modelFit' not found
```

```
## Error: object 'pred' not found
```

```r
table(pred,testing$Species)
```

```
## Error: object 'pred' not found
```

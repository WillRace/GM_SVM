package.loader <- function(pack, library = F){
  checkInstall <- pack[!(pack %in% installed.packages()[, "Package"])]
  if(length(checkInstall))
    install.packages(checkInstall, dependencies = T)
  if(library == T)
    sapply(pack, require, character.only = T)
  }

package.loader("mlr", library = T)
package.loader(c("caret"))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                                                     #
# A function for performing cross validated SVM modeling on GM data, with PCA feature extraction as a #
# pre-processing step. Can be applied to a dataset for 'Leave One Out' CV through use of the 'lapply' #
# function given a vector of integers corresponding to test individuals eg:                           #
#                                                                                                     #
# testers <- c(1:90); lapply(testers, gm.pca.SVM, data = svmData, pcSelect = c(1:10), plotSVM = T)    #
#                                                                                                     #
# GM data should be provided in the form of a dataframe where the first variable is a class variable  #
# and subsequent variables are numeric (centroid size and procrustes shape coordinates).              #
#                                                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

gm.SVM <- function(i, data){
  
  set.seed(123)
  
  print(paste("Sampling ID:", i, "from data"))
  
  ### Pre-processing ###
  
  # Train/Test split #
  
  train <- data[-i,] # Removes test items from training set
  test <- data[i,-1] # Creates a new dataset containing test items without labels
  
  # Normalise Data #
  
  for(n in names(train[,-1])){
    test[,n] <- (test[,n] - min(train[,n])) / (max(train[,n]) - min(train[,n]))
    train[,n] <- (train[,n] - min(train[,n])) / (max(train[,n]) - min(train[,n]))
  }
  
  ### SVM Section ###
  
  # Hyperparameter Tuning #
  
  svmTask <- makeClassifTask(data = train, 
                             target = "target")
  
  svmLearner <- makeLearner("classif.svm", predict.type = "response")
  
  svmResample <- makeResampleDesc("RepCV", reps = 5, folds = 5, stratify = T)

  svmControl <- makeTuneControlGrid(resolution = 10L)
  
  svmParams <- makeParamSet(
    makeDiscreteParam("kernel", values = "linear"),
    makeNumericParam("cost", lower = -5, upper = 5, trafo = function(x) 2^x))
  
  tunedSVMParams <- tuneParams("classif.svm", task = svmTask, 
                               resampling = svmResample,
                               par.set = svmParams,
                               control = svmControl,
                               measures = list(f1))
  
  tunedSVM <- setHyperPars(svmLearner, par.vals = tunedSVMParams$x)
  
  # Training and testing #
  
  modelSVM <- train(tunedSVM, svmTask)
  
  testSVM <- predict(object = modelSVM, newdata = test)
  
  ### Function Returns ###
  
  return(list("Observed Genus" = data[i,1],
              "SVM Result" = as.character(testSVM[["data"]][['response']]),
              "test.train" = list("train" = train,
                                  "test" = test)))
}


# LOO Cross Validation -----------------------------------------------------------------------------

cross.validate <- function(data){
  
  parallelMap::parallelStartSocket(cpus = parallel::detectCores())
  
  testers <- c(1:nrow(data))
  
  results <- lapply(testers, 
                    gm.SVM, 
                    data = data)
  
  parallelMap::parallelStop()
  
  results <- data.frame("Actual Class" = sapply(results, '[[', 1),
                        "Predicted Class" = sapply(results, '[[', 2))
  
  caretMatrix <- caret::confusionMatrix(table(resultsLM1[,1], 
                                              resultsLM1[,2]))
  
  return(list("Result" = results,
              "Caret Stats" = caretMatrix))
}
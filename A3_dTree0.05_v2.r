# Classification Tree with rpart. 
# Adapted from https://www.statmethods.net/advstats/cart.html

# First, clear all previous stuff out of the workspace...
rm(list = ls())

# Loading Packages and Import Data
# Check for, and install if needed, these packages:
# rpart      - decision tree module
# rpart.plot - gives us the fancy decision tree plot

if (!require("rpart")) { install.packages("rpart")
  require("rpart") }

if (!require("rpart.plot")) { install.packages("rpart.plot")
  require("rpart.plot") }

# Import dataset from the file Titanic.csv
# Make sure your data is saved in your Working Directory folder. 
# How to set your working directory folder? "Session"- "Set Working Directory" - "Choose Directory"
dataSet = read.csv("BankLoan2.csv")

# Read the data file. Make sure:
#     1) The first row contain the data labels
#     2) Numeric data fields only contain numbers
#     3) The order of the data match the order of the labels
#     4) Each observation is numbered with an identification number called ID


# This just rigs the random number generator so it always comes out the same.
# It keeps out results consistent over multiple runs.
set.seed(123) 

#######################################################################
######  Partitioning the data into Training and Validation Sets  ######  

#      a training set (used to build the tree), and
#      a validation set (used to verify how good the tree is). 
# sample() is a function that takes a sample of the specific size (using the second parameter) 
#          from the data (specified using the first parameter)
# 1:nrow(dataSet) is the full sample (a vector of row indices) that we will sample from
# size = round(TRAINING_PART*nrow(dataSet)) tells R what proportion (TRAINING_PART) of the data to assign to the (test) parition
# replace = F      tells R to sample from the full set without replacement
# so we are essentially sampling row indices with the following command:
trainIndex  <- sample(1:nrow(dataSet), size = round(0.5*nrow(dataSet)), replace = F)

# 0.5 means 50% of the data were assigned to the training set

# Put the subset of data assigned as the training partition (specified by p) in trainingSet
trainingSet <- dataSet[ trainIndex,]

# Put everything that wasn't in that subset (1-p) in validationSet
validationSet  <- dataSet[-trainIndex,]

#######################################################################
######       Creating a decision tree model called MyTree        ######  

# It uses the trainingSet as its data source.
# If you want to create a different decision tree model, you'll need to change the first parameter:
#    Survived ~ Pclass + Sex + Age + SibSp+ Parch + Fare
# The first term (Survived) is the outcome (here, 1 = survive, 0=don't survive)
# and the rest (Pclass, Sex, Age, etc) are the potential inputs.Pclass indicates each passenger's class (1st, 2nd, or 3rd)
#   SibSp indicates the number of siblings/spouses aboard the Titanic, Parch indicates the number of parents/children aboard the Titanic
#   Fare is the fare paid for ticket
# If you use a different dataset, or want to try different inputs, you'll need to change the model!!
MyTree <- rpart(payback ~ age + sex +	region +	income + married + children + car + save_act + current_act + mortgage, data=trainingSet, method="class", 
                control=rpart.control(minsplit=25, cp=0.05))


# In the above model, minsplit defines the minimum number of observations in each node needed to add an additional split.
#                         (larger number = simpler, but maybe less accurate tree)
# cp defines the minimum reduction in error needed to add an additional split.
#                         (larger number = simpler, but maybe less accurate tree) 
# Here I choose 10 and 0.005. You can use different values if you want a different tree.

# Turn on output to a file (in addition to the screen). This way we've got a record of
# what we did.
#   append=FALSE means overwrite the file if it already exists
#   split=TRUE   means send the output to the console too!
sink("DecisionTreeOuput.txt", append=FALSE, split=TRUE)

# Display the text output from each decision tree. 
# You're looking for the tree that minimizes xerror. 
# xerror is the relative misclassification rate compared to the "no split" tree (lowest = best).
cat("\n###### Display the text output from each decision tree: ######\n")
printcp(MyTree)

# Show the result graphically
plotcp(MyTree, minline = FALSE)

#######################################################################
######                     Pruning the tree                      ######
# This builds a modified version of the original tree
#     which uses the splits that generate the lowest xerror.
prunedTree <- prune(MyTree, cp=MyTree$cptable[which.min(MyTree$cptable[,"xerror"]),"CP"])

#######################################################################
######          Evaluating Classification Accuracy               ######
# predict() gives us the predicted values for each observation in the two sets 
# ...given the prunedTree model.
predTraining <- predict(MyTree, trainingSet, type="class") 
predValidation <- predict(MyTree, validationSet, type="class")

# Generating Confusion Matrices for the traing and validation sets:
cat("\n###### Confusion Matrix for the training set ######\n")
table(Predicted=predTraining,Observed=trainingSet[, 12] )

# trainingSet[, 2] tells R the 2nd column of the dataset has the outcome variable (Survived)
# If your outcome variable is in a different column, please change the number "2" accordingly

cat("\n###### Confusion Matrix for the validation set ######\n")
table(Predicted=predValidation,Observed=validationSet[, 12] )
# validationSet[, 2] tells R the 2nd column of the dataset has the outcome variable (Survived)
# If your outcome variable is in a different column, please change the number "2" accordingly

# Correct Classification Rate:
# Check whether there is a match between each predicted value (in pred) and the actual value
#    (The value of OUTCOME_COL is specified above and is the location in the data file where the actual outcome values are stored.)
# If it's a match, that's TRUE ("1"); if it's not a match, that's FALSE ("0").
# So if you take the mean it will give you the proportion of times it was a match (i.e,. correct).
#    Imagine five predictions, 3 were right, 2 were wrong (1, 1, 0, 1, 0). So the average = 3/5 = 60%.
predRateTraining <- mean(predTraining == trainingSet[, 12])
predRateValidation <- mean(predValidation == validationSet[, 12])

# This stops R from writing any more to the text output file.
sink()

# Now plot the pruned tree, which is all we really care about to make our decisions.
# Don't touch any of this stuff!! It's set up to work with whatever dataset you're using!!
# HOW TO READ THE TREE:
#     Each branch will have split criteria. Look for the branch with the right value for your case.
#     The number beneath each node is the probability of "survive" (i.e., an outcome of "1")
#     IGNORE the 0s and 1s inside the nodes. This is just numbering each branch from the parent node.
# 
# What's up with paste()?
#     This function just puts things together. The "main" parameter is the title for the chart.
#     paste builds the text in the label by putting together three things (separated by commas):
#     #1: Decision Tree
#         (Classifies Correctly 
#     #2: A percentage, rounded to two decimal places (the proportion rounded to four places then x 100)
#     #3: % of the time)
#     When you string them all together you get a nicely formatted title, and your friends are impressed!
prp(prunedTree, main=paste("Decision Tree\n(Correct classification rate ",
                           round(predRateTraining,4)*100,
                           "% for the training set\n ",
                           round(predRateValidation,4)*100,
                           "% for the validation set)"), 
    type=4, extra=6, faclen=0, under=TRUE)

# And now turn on PDF output to send the same plot to a PDF file, saved to the working directory.
pdf("TreeOutput.pdf");
prp(prunedTree, main=paste("Decision Tree\n(Correct classification rate ",
                           round(predRateTraining,4)*100,
                           "% for the training set\n ",
                           round(predRateValidation,4)*100,
                           "% for the validation set)"), 
    type=4, extra=6, faclen=0, under=TRUE)
dev.off()


















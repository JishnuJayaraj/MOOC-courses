# -*- coding: utf-8 -*-
"""
Created on Tue May 28 00:13:58 2019

@author: test
https://github.com/justmarkham/scikit-learn-videos

What is the purpose of model evaluation, and what are some common evaluation procedures?
What is the usage of classification accuracy, and what are its limitations?
How does a confusion matrix describe the performance of a classifier?
What metrics can be computed from a confusion matrix?
How can you adjust classifier performance by changing the classification threshold?
What is the purpose of an ROC curve?
How does Area Under the Curve (AUC) differ from classification accuracy?


Model evaluation procedures-Model evaluation metrics-
"""


#%%
# read the data into a pandas DataFrame
import pandas as pd
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(path, header=None, names=col_names)

# print the first 5 rows of data
print(pima.head())

#Can we predict the diabetes status of a patient given their health measurements?

# define X and y
feature_cols = ['pregnant', 'insulin', 'bmi', 'age']
X = pima[feature_cols]
y = pima.label

# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# make class predictions for the testing set
y_pred_class = logreg.predict(X_test)



# calculate accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class))


#null accuracy
# examine the class distribution of the testing set (using a Pandas Series method)
y_test.value_counts()

# calculate the percentage of ones
y_test.mean()

# calculate the percentage of zeros
1 - y_test.mean()

# calculate null accuracy (for binary classification problems coded as 0/1)
max(y_test.mean(), 1 - y_test.mean())


# calculate null accuracy (for multi-class classification problems)
y_test.value_counts().head(1) / len(y_test)

# print the first 25 true and predicted responses
print('True:', y_test.values[0:25])
print('Pred:', y_pred_class[0:25])
#%%

#Confusion matrix
#%%
# IMPORTANT: first argument is true values, second argument is predicted values
print(metrics.confusion_matrix(y_test, y_pred_class))

# print the first 25 true and predicted responses
print('True:', y_test.values[0:25])
print('Pred:', y_pred_class[0:25])


# save confusion matrix and slice into four pieces
confusion = metrics.confusion_matrix(y_test, y_pred_class)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

#Classification Accuracy: Overall, how often is the classifier correct?
print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, y_pred_class))

#Classification Error: Overall, how often is the classifier incorrect?
#Also known as "Misclassification Rate"
print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, y_pred_class))

#Sensitivity: When the actual value is positive, how often is the prediction correct?
#How "sensitive" is the classifier to detecting positive instances?
#Also known as "True Positive Rate" or "Recall"
print(TP / float(TP + FN))
print(metrics.recall_score(y_test, y_pred_class))


#Specificity: When the actual value is negative, how often is the prediction correct?
#How "specific" (or "selective") is the classifier in predicting positive instances?
print(TN / float(TN + FP))

#False Positive Rate: When the actual value is negative, how often is the prediction incorrect?
print(FP / float(TN + FP))


#Precision: When a positive value is predicted, how often is the prediction correct?
#How "precise" is the classifier when predicting positive instances?
print(TP / float(TP + FP))
print(metrics.precision_score(y_test, y_pred_class))




#%%

#Adjusting the classification threshold...
#%%
# print the first 10 predicted responses
logreg.predict(X_test)[0:10]

# print the first 10 predicted probabilities of class membership
logreg.predict_proba(X_test)[0:10, :]

# print the first 10 predicted probabilities for class 1
logreg.predict_proba(X_test)[0:10, 1]

# store the predicted probabilities for class 1
y_pred_prob = logreg.predict_proba(X_test)[:, 1]

import matplotlib.pyplot as plt

# histogram of predicted probabilities
plt.hist(y_pred_prob, bins=8)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of diabetes')
plt.ylabel('Frequency')



#Decrease the threshold for predicting diabetes in order to increase the sensitivity of the classifier


# predict diabetes if the predicted probability is greater than 0.3
from sklearn.preprocessing import binarize
y_pred_class = binarize([y_pred_prob], 0.3)[0]

# print the first 10 predicted probabilities
y_pred_prob[0:10]

# print the first 10 predicted classes with the lower threshold
y_pred_class[0:10]

# previous confusion matrix (default threshold of 0.5)
print(confusion)

# new confusion matrix (threshold of 0.3)
print(metrics.confusion_matrix(y_test, y_pred_class))

# sensitivity has increased (used to be 0.24)
print(46 / float(46 + 16))

# specificity has decreased (used to be 0.91)
print(80 / float(80 + 50))


#%%


#ROC Curves and Area Under the Curve 
#%%
#Wouldn't it be nice if we could see how sensitivity and specificity are affected by various thresholds, without actually changing the threshold?
# IMPORTANT: first argument is true values, second argument is predicted probabilities

import matplotlib.pyplot as plt
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# define a function that accepts a threshold and prints sensitivity and specificity
def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])
    
evaluate_threshold(0.5)

evaluate_threshold(0.3)

# IMPORTANT: first argument is true values, second argument is predicted probabilities
print(metrics.roc_auc_score(y_test, y_pred_prob))

# calculate cross-validated AUC
from sklearn.model_selection import cross_val_score
cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()


#%%
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 14:40:03 2019

@author: test

Cross-validation for parameter tuning, model selection, and feature selection 

Video 7, Lesson7
https://www.youtube.com/watch?v=6dbrR-WymjI&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=7
https://github.com/justmarkham/scikit-learn-videos/blob/master/07_cross_validation.ipynb

What is the drawback of using the train/test split procedure for model evaluation?
How does K-fold cross-validation overcome this limitation?
How can cross-validation be used for selecting tuning parameters, choosing between models, and selecting features?
What are some possible improvements to cross-validation?

"""
#%%
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# read in the iris data
iris = load_iris()

# create X (features) and y (response)
X = iris.data
y = iris.target


# use train/test split with different random_state values, 2 gives best
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)

# check classification accuracy of KNN with K=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


#%%

# Steps for K-fold cross-validation..

#%%
# simulate splitting a dataset of 25 observations into 5 folds
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=False).split(range(25))

# print the contents of each training and testing set
print('{} {:^61} {}'.format('Iteration', 'Training set observations', 'Testing set observations'))
for iteration, data in enumerate(kf, start=1):
    print('{:^9} {} {:^25}'.format(iteration, data[0], str(data[1])))

#%%


#%%
# Cross-validation example: parameter tuning
    
from sklearn.model_selection import cross_val_score


# 10-fold cross-validation with K=5 for KNN (the n_neighbors parameter)
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores)


# use average accuracy as an estimate of out-of-sample accuracy
print(scores.mean())

#%%


# search for an optimal value of K for KNN
#%%
k_range = list(range(1, 31))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)


import matplotlib.pyplot as plt
%matplotlib inline

# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
#%%

# Cross-validation example: model selection
#%%
#-->1
# 10-fold cross-validation with the best KNN model
knn = KNeighborsClassifier(n_neighbors=20)
print(cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean())

#-->2
# 10-fold cross-validation with logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
print(cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean())

# comparing... knn is best
#%%

#Cross-validation example: feature selection
#Goal: Select whether the Newspaper feature should be included in the linear regression model on the advertising dataset
#%%

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# read in the advertising dataset
data = pd.read_csv('Advertising.csv', index_col=0)

# create a Python list of three feature names
feature_cols = ['TV', 'radio', 'newspaper']

# use the list to select a subset of the DataFrame (X)
X = data[feature_cols]

# select the Sales column as the response (y)
y = data.sales



# 10-fold cross-validation with all three features
lm = LinearRegression()
scores = cross_val_score(lm, X, y, cv=10, scoring='neg_mean_squared_error')
print(scores)


# fix the sign of MSE scores
mse_scores = -scores
print(mse_scores)

# convert from MSE to RMSE
rmse_scores = np.sqrt(mse_scores)
print(rmse_scores)

# calculate the average RMSE
print(rmse_scores.mean())





# 10-fold cross-validation with two features (excluding Newspaper)
feature_cols = ['TV', 'radio']
X = data[feature_cols]
print(np.sqrt(-cross_val_score(lm, X, y, cv=10, scoring='neg_mean_squared_error')).mean())
#%%


# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:34:33 2019

@author: test
Video 8, Lesson 8
https://github.com/justmarkham/scikit-learn-videos/blob/master/08_grid_search.ipynb

How can K-fold cross-validation be used to search for an optimal tuning parameter?
How can this process be made more efficient?
How do you search for multiple tuning parameters at once?
What do you do with those tuning parameters before making real predictions?
How can the computational expense of this process be reduced?

"""
#More efficient parameter tuning using GridSearchCV
#Allows you to define a grid of parameters that will be searched using K-fold cross-validation
#%%
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV

# read in the iris data
iris = load_iris()

# create X (features) and y (response)
X = iris.data
y = iris.target

knn = KNeighborsClassifier()
# define the parameter values that should be searched
k_range = list(range(1, 31))
print(k_range)

# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors=k_range)
print(param_grid)

# instantiate the grid
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False)


# fit the grid with data
grid.fit(X, y)

# view the results as a pandas DataFrame
import pandas as pd
pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]

# examine the first result
print(grid.cv_results_['params'][0])
print(grid.cv_results_['mean_test_score'][0])

# print the array of mean scores only
grid_mean_scores = grid.cv_results_['mean_test_score']
print(grid_mean_scores)

# plot the results
plt.plot(k_range, grid_mean_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')

# examine the best model
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)
#%%

'''
Example: tuning max_depth and min_samples_leaf for a DecisionTreeClassifier
Could tune parameters independently: change max_depth while leaving min_samples_leaf at its default value, and vice versa
But, best performance might be achieved when neither parameter is at its default value
'''
#Searching multiple parameters simultaneously
#%%

# define the parameter values that should be searched
k_range = list(range(1, 31))
weight_options = ['uniform', 'distance']

# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors=k_range, weights=weight_options)
print(param_grid)

# instantiate and fit the grid
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False)
grid.fit(X, y)

# view the results
pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]

# examine the best model
print(grid.best_score_)
print(grid.best_params_)


#%%

#Using the best parameters to make predictions
#%%

# train your model using all data and the best known parameters
knn = KNeighborsClassifier(n_neighbors=13, weights='uniform')
knn.fit(X, y)

# make a prediction on out-of-sample data
knn.predict([[3, 5, 4, 2]])

# shortcut: GridSearchCV automatically refits the best model using all of the data
grid.predict([[3, 5, 4, 2]])

#%%

#Reducing computational expense using RandomizedSearchCV
#%%
#RandomizedSearchCV searches a subset of the parameters, and you control the computational "budget"

from sklearn.model_selection import RandomizedSearchCV

# specify "parameter distributions" rather than a "parameter grid"
param_dist = dict(n_neighbors=k_range, weights=weight_options)

# n_iter controls the number of searches
rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=10, random_state=5, return_train_score=False)
rand.fit(X, y)
pd.DataFrame(rand.cv_results_)[['mean_test_score', 'std_test_score', 'params']]

# examine the best model
print(rand.best_score_)
print(rand.best_params_)


# run RandomizedSearchCV 20 times (with n_iter=10) and record the best score
best_scores = []
for _ in range(20):
    rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=10, return_train_score=False)
    rand.fit(X, y)
    best_scores.append(round(rand.best_score_, 3))
print(best_scores)
#%%
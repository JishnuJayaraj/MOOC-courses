# -*- coding: utf-8 -*-
"""
Created on Wed May 15 16:25:45 2019

@author: test

other data sets: https://scikit-learn.org/stable/datasets/
"""

'''

Requirements for working with data in scikit-learn
Features and response are separate objects
Features and response should be numeric
Features and response should be NumPy arrays
Features and response should have specific shapes

'''




# import load_iris function from datasets module
from sklearn.datasets import load_iris

# save "bunch" object containing iris dataset and its attributes
iris = load_iris()
type(iris)

print (type(iris))

# print the iris data
print(iris.data)
'''
Each row is an /observation/ (also known as: sample, example, instance, record)
Each column is a /feature/ (also known as: predictor, attribute, independent variable, input, regressor, covariate)
'''


# print the names of the four features
print(iris.feature_names)

# print integers representing the species of each observation
print(iris.target)

# print the encoding scheme for species: 0 = setosa, 1 = versicolor, 2 = virginica
print(iris.target_names)



#-------------------------------------------------------------------------


'''
150 observations
4 features (sepal length, sepal width, petal length, petal width)
Response variable is the iris species
Classification problem since response is categorical
'''
# store feature matrix in "X"
X = iris.data

# store response vector in "y"
y = iris.target

# print the shapes of X and y
print(X.shape)
print(y.shape)

# 4 STEP MODELLING PATTERN


# Step 1: Import the class you plan to use
# https://scikit-learn.org/stable/modules/neighbors.html
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
from sklearn.neighbors import KNeighborsClassifier

# Step 2: "Instantiate" the "estimator"

# "Estimator" is scikit-learn's term for model
# "Instantiate" means "make an instance of"
knn = KNeighborsClassifier(n_neighbors=1)
print(knn)


# Step 3: Fit the model with data (aka "model training")

# Model is learning the relationship between X and y
# Occurs in-place
knn.fit(X, y)

# Step 4: Predict the response for a new observation

# New observations are called "out-of-sample" data
# Uses the information it learned during the model training process
knn.predict([[3, 5, 4, 2]])
# Returns a NumPy array, a([2])  means virgenia
# Can predict for multiple observations at once
X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
knn.predict(X_new)

'''

# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X, y)

# predict the response for new observations
logreg.predict(X_new)
'''

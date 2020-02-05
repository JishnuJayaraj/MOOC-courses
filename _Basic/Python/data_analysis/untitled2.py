# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:47:14 2019

@author: test
week4: Model development


"""
# develop several models that will predict the price of the car using the variables or features.

#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# path of data 
path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)
df.head()

# -- Linear regression
from sklearn.linear_model import LinearRegression

# create linear regression obj
lm = LinearRegression()
lm

# Highway-mpg help us predict car price
X = df[['highway-mpg']]
Y = df['price']

lm.fit(X,Y)

Yhat=lm.predict(X)
Yhat[0:5]   

# value of the intercept (a)
lm.intercept_

# value of the Slope (b)
lm.coef_



# --Multiple Linear Regression

#if we want to predict car price using more than one variable

Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]

#Fit the linear model using the four above-mentioned variables.
lm.fit(Z, df['price'])

lm.intercept_
lm.coef_
# 𝑌ℎ𝑎𝑡=𝑎+𝑏1𝑋1+𝑏2𝑋2+b3X3+b4X4


#%%



#%%
# developed some models, how do we evaluate our models and how do we choose the best one?

# import the visualization package: seaborn
import seaborn as sns

# --simple linear regression, an excellent way to visualize the fit of our model is by using regression plots.

width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)

# how scattered the data points are around the regression line. This will give you a good indication of the variance of the data, and whether a linear model would be the best fit or not. If the data is too far off from the line, this linear model might not be the best model for this data

# compare this plot to the regression plot of "peak-rpm".
plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)
#points for "highway-mpg" are much closer to the generated line and on the average decrease



# -- A good way to visualize the variance of the data is to use a residual plot.

#A residual plot is a graph that shows the residuals on the vertical y-axis and the independent variable on the horizontal x-axis

# If the points in a residual plot are randomly spread out around the x-axis, then a linear model is appropriate for the data. 
# Randomly spread out residuals means that the variance is constant, and thus the linear model is a good fit for this data.

width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(df['highway-mpg'], df['price'])
plt.show()
# are not randomly spread around the x-axis, which leads us to believe that maybe a non-linear model is more appropriate for this data.









#%%

# --------------------------- Visualize Multiple lnear regerssion
#%%
# look at the distribution of the fitted values that result from the model and compare it to the distribution of the actual values.

Y_hat = lm.predict(Z)

plt.figure(figsize=(width, height))


ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)


plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()
#fitted values are reasonably close to the actual values, since the two distributions overlap a bit. However, there is definitely some room for improvement.



#%%

# ------------------------Polynomial regerssion
#%%
#particular case of the general linear regression model or multiple linear regression models
# are different orders of polynomial regression:
#We saw earlier that a linear model did not provide the best fit while using highway-mpg as the predictor variable. Let's see if we can try fitting a polynomial model to the data instead.

def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()  # ??
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()
    
x = df['highway-mpg']
y = df['price']

# Here we use a polynomial of the 3rd order (cubic) 

# fit the polynomial using the function polyfit, then use the function poly1d to display the polynomial function.

f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)

PlotPolly(p, x, y, 'highway-mpg')

np.polyfit(x, y, 3)

#-- We can perform a polynomial transform on multiple features
from sklearn.preprocessing import PolynomialFeatures

#create a PolynomialFeatures object of degree 2
pr=PolynomialFeatures(degree=2)
pr

Z_pr=pr.fit_transform(Z)

Z.shape
Z_pr.shape



#%%

# ----------------------------Pipeline
#%%
# simplify the steps of processing the data. We use the module Pipeline to create a pipeline. We also use StandardScaler as a step in our pipeline.

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#We create the pipeline, by creating a list of tuples including the name of the model or estimator and its corresponding constructor
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]

# we input the list as an argument to the pipeline constructor
pipe=Pipeline(Input)
pipe

# We can normalize the data, perform a transform and fit the model simultaneously
pipe.fit(Z,y)

# Similarly, we can normalize the data, perform a transform and produce a prediction simultaneously
ypipe=pipe.predict(Z)
ypipe[0:4]






#%%



# ---------------------------Measures for In-Sample Evaluation

#%%
# quantitative measure to determine how accurate the model is.
# R^2: coefficient of determination, is a measure to indicate how close the data is to the fitted regression line.
# MSE: Mean Squared Error measures the average of the squares of errors,

# Let's calculate the R^2
#==1 highway_mpg_fit
lm.fit(X, Y)
# Find the R^2
print('The R-square is: ', lm.score(X, Y))

# calculate the MSE
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)

#==2 Multiple linear regression

# fit the model 
lm.fit(Z, df['price'])
# Find the R^2
print('The R-square is: ', lm.score(Z, df['price']))

Y_predict_multifit = lm.predict(Z)
print('The mean square error of price and predicted value using multifit is: ', \
      mean_squared_error(df['price'], Y_predict_multifit))


# ==3 polynomial fit
from sklearn.metrics import r2_score
r_squared = r2_score(y, p(x))
print('The R-square value is: ', r_squared)

mean_squared_error(df['price'], p(x))



#%%

# ---------------- Prediction and decision making

#%%

#Create a new input
new_input=np.arange(1, 100, 1).reshape(-1, 1)


#Fit the model
lm.fit(X, Y)

#Produce a prediction
yhat=lm.predict(new_input)
yhat[0:5]

#we can plot the data
plt.plot(new_input, yhat)
plt.show()


#%%
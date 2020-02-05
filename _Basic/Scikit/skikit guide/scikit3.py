'''
https://github.com/justmarkham/scikit-learn-videos/blob/master/06_linear_regression.ipynb
https://www.youtube.com/watch?v=3ZWuPVWq7p4&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=7&t=182s

How do I use the pandas library to read data into Python?
How do I use the seaborn library to visualize data?
What is linear regression, and how does it work?
How do I train and interpret a linear regression model in scikit-learn?
What are some evaluation metrics for regression problems?
How do I choose which features to include in my model?

'''

# Regression

# conventional way to import pandas
import pandas as pd

# read CSV file from the 'data' subdirectory using a relative path
data = pd.read_csv('Advertising.csv')

# display the first 5 rows
print(data.head())

'''
Primary object types:

DataFrame: rows and columns (like a spreadsheet)
Series   : a single column
'''
# display the last 5 rows
# print(data.tail())

# setting index column
data2=pd.read_csv('Advertising.csv', index_col=0)
print(data2.head())

# check the shape of the DataFrame (rows, columns)
print(data2.shape)


'''
What are the features?

TV: advertising dollars spent on TV for a single product in a given market (in thousands of dollars)
Radio: advertising dollars spent on Radio
Newspaper: advertising dollars spent on Newspaper
What is the response?

Sales: sales of a single product in a given market (in thousands of items)
What else do we know?

Because the response variable is continuous, this is a regression problem.
There are 200 observations (represented by the rows), and each observation is a single market.
'''

# Seaborn: Python library for statistical data visualization built on top of Matplotlib

# conventional way to import seaborn
import seaborn as sns

# visualize the relationship between the features and the response using scatterplots
sns.pairplot(data, x_vars=['TV','radio','newspaper'], y_vars='sales', height=7, aspect=0.7)

# fit a linear regression
sns.pairplot(data, x_vars=['TV','radio','newspaper'], y_vars='sales', height=7, aspect=0.7, kind='reg')

'''
Linear regression
Pros: fast, no tuning required, highly interpretable, well-understood

Cons: unlikely to produce the best predictive accuracy (presumes a linear relationship between the features and response)
further details in link...

Preparing X and y using pandas
scikit-learn expects X (feature matrix) and y (response vector) to be NumPy arrays.
However, pandas is built on top of NumPy.
Thus, X can be a pandas DataFrame and y can be a pandas Series!
'''


# create a Python list of feature names
feature_cols = ['TV', 'radio', 'newspaper']

# use the list to select a subset of the original DataFrame
X = data[feature_cols]

# equivalent command to do this in one line
# X = data[['TV', 'Radio', 'Newspaper']]

# print the first 5 rows
print(X.head())
# check the type and shape of X
print(type(X))
print(X.shape)

# select a Series from the DataFrame
y = data['sales']

# equivalent command that works if there are no spaces in the column name
# y = data.Sales

# print the first 5 values
print(y.head())
# check the type and shape of y
print(type(y))
print(y.shape)



'''
Splitting X and y into training and testing sets
'''


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# default split is 75% for training and 25% for testing
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


''' LINEAR REGRESSION'''
# import model
from sklearn.linear_model import LinearRegression

# instantiate
linreg = LinearRegression()

# fit the model to the training data (learn the coefficients)
linreg.fit(X_train, y_train)



# ----- Interpreting model coefficients ----

# print the intercept and coefficients
print(linreg.intercept_)
print(linreg.coef_)

# pair the feature names with the coefficients
list(zip(feature_cols, linreg.coef_))

# y = 2.88 + 0.0466* TV + 0.179* Radio + 0.00345 * Newspaper
# How do we interpret the TV coefficient (0.0466)?

# For a given amount of Radio and Newspaper ad spending, a "unit" increase in TV ad spending is associated with a 0.0466 "unit" increase in Sales.
# Or more clearly: For a given amount of Radio and Newspaper ad spending, an additional $1,000 spent on TV ads is associated with an increase in sales of 46.6 items.


# ----- making prediction ----
# make predictions on the testing set
y_pred = linreg.predict(X_test)

'''
We need an evaluation metric in order to compare our predictions with the actual values!
Model evaluation metrics for regression¶
Evaluation metrics for classification problems, such as accuracy, are not useful for regression problems. Instead, we need evaluation metrics designed for comparing continuous values.

Let's create some example numeric predictions, and calculate three common evaluation metrics for regression problems:
    
    '''

# define true and predicted response values
true = [100, 50, 30, 20]
pred = [90, 50, 50, 30]


# ---1) calculate MAE by hand, Mean Absolute Error
print((10 + 0 + 20 + 10)/4.)

# calculate MAE using scikit-learn
from sklearn import metrics
print(metrics.mean_absolute_error(true, pred))


#--2) calculate MSE by hand
print((10**2 + 0**2 + 20**2 + 10**2)/4.)

# calculate MSE using scikit-learn
print(metrics.mean_squared_error(true, pred))


# ---3) calculate RMSE by hand
import numpy as np
print(np.sqrt((10**2 + 0**2 + 20**2 + 10**2)/4.))

# calculate RMSE using scikit-learn
print(np.sqrt(metrics.mean_squared_error(true, pred)))

'''

Comparing these metrics:

MAE is the easiest to understand, because it's the average error.
MSE is more popular than MAE, because MSE "punishes" larger errors.
RMSE is even more popular than MSE, because RMSE is interpretable in the "y" units.
'''


# Computing the RMSE for our Sales predictions
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
 
'''
 
Feature selection
Does Newspaper "belong" in our model? In other words, does it improve the quality of our predictions?

Let's remove it from the model and check the RMSE!
...
'''
 
 
 
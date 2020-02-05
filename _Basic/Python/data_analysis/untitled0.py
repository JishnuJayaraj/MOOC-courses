# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 12:02:31 2019

@author: test
"""
#%%

# import pandas library
import pandas as pd

# Read the online file by the URL provides above, and assign it to variable "df"
other_path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"
df = pd.read_csv(other_path, header=None)

# show the first 5 rows using dataframe.head() method
print("The first 5 rows of the dataframe") 
print(df.head(5))

# create headers list
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
print("headers\n", headers)

df.columns = headers
df.head(10)

#we can drop missing values along the column "price" as follows
df.dropna(subset=["price"], axis=0)


# Data has a variety of types.
# The main types stored in Pandas dataframes are object, float, int, bool and datetime64.


# check the data type of data frame "df" by .dtypes
print(df.dtypes)

# If we would like to get a statistical summary of each column, such as count, column mean value, column standard deviation, etc. We use the describe method:
print(df.describe())

#  provides the statistical summary of all the columns, including object-typed attributes.
# describe all the columns in "df" 
df.describe(include = "all")

# select the columns of a data frame by indicating the name of each column
df1 = df[['length','width', 'height']]
df1.describe()

# look at the info of "df"
df.info
#%%


#%%
#---------------------------------WEEK 2 ---------------------------
#Data Wrangling is the process of converting data from the initial format to a format that may be better for analysis.
# https://labs.cognitiveclass.ai/tools/jupyterlab/lab/tree/labs/DA0101EN/data-wrangling.ipynb
 

import pandas as pd
import matplotlib.pylab as plt

filename = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df = pd.read_csv(filename, names = headers)

# To see what the data set looks like, we'll use the head() method.
df.head()
# see many ? in data

#We replace "?" with NaN (Not a Number), which is Python's default missing value marker, for reasons of computational speed and convenience.
import numpy as np

#-- replace "?" to NaN
df.replace("?", np.nan, inplace = True)
df.head(5)



#-- Evaluating for Missing Data
missing_data = df.isnull()
missing_data.head(5)
# true: missing, false: not, *also isnotnull()

# --Count missing values in each column
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("") 

# value_counts(): counts the number of "True" values.
    

# ------ Deal with missing data
    
# -- Calc average of the column
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)

# --Replace "NaN" by mean value in "normalized-losses" column
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)

#To see which values are present in a particular column, we can use the ".value_counts()" method:
df['num-of-doors'].value_counts()

#We can also use the ".idxmax()" method to calculate for us the most common type automatically:
df['num-of-doors'].value_counts().idxmax()
#ans is 4

#replace the missing 'num-of-doors' values by the most frequent 
df["num-of-doors"].replace(np.nan, "four", inplace=True)

# --Finally, let's drop all rows that do not have price data:
# simply drop whole row with NaN in "price" column
df.dropna(subset=["price"], axis=0, inplace=True)

# reset index, because we droped two rows
df.reset_index(drop=True, inplace=True)

df.head()
# dataset with no missing data

# ---------------------CORRECT DATA FORMAT

#.dtype() to check the data type
#.astype() to change the data type

#Lets list the data types for each column
df.dtypes

#-- Convert data types to proper format
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

df.dtypes


# ---------------------- Data Standardization/ normalisation
#process of transforming data into a common format which allows the researcher to make the meaningful comparison.

# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]

# check your transformed data 
df.head()


# replace (original value) by (original value)/(maximum value)
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()



# ------- BINNING
# transforming continuous numerical variables into discrete categorical 'bins', for grouped analysis.

# Convert data to correct format
df["horsepower"]=df["horsepower"].astype(int, copy=True)

#plot the histogram of horspower, to see what the distribution of horsepower looks like.


plt.pyplot.hist(df["horsepower"])

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


#We would like 3 bins of equal size bandwidth so we use numpy's linspace(start_value, end_value, numbers_generated function.
#Since we want to include the minimum value of horsepower we want to set start_value=min(df["horsepower"]).

#Since we want to include the maximum value of horsepower we want to set end_value=max(df["horsepower"]).

#Since we are building 3 bins of equal length, there should be 4 dividers, so numbers_generated=4.

bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
bins

# We set group names:
group_names = ['Low', 'Medium', 'High']


# We apply the function "cut" the determine what each value of "df['horsepower']" belongs to.
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
df[['horsepower','horsepower-binned']].head(20)

# Lets see the number of vehicles in each bin
df["horsepower-binned"].value_counts()

from matplotlib import pyplot
pyplot.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


#---- Bin visualisation

a = (0,1,2)

# draw historgram of attribute "horsepower" with bins = 3
plt.pyplot.hist(df["horsepower"], bins = 3)

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")



# ------ Indicator/ ummy variable
# numerical variable used to label categories. They are called 'dummies' because the numbers themselves don't have inherent meaning.

# We will use the panda's method 'get_dummies' to assign numerical values to different categories of fuel type.
df.columns

dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.head()


dummy_variable_1.rename(columns={'fuel-type-diesel':'gas', 'fuel-type-diesel':'diesel'}, inplace=True)
dummy_variable_1.head()

# We now have the value 0 to represent "gas" and 1 to represent "diesel" in the column "fuel-type". We will now insert this column back into our original dataset.

# merge data frame "df" and "dummy_variable_1" 
df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)

df.head()
#The last two columns are now the indicator variable representation of the fuel-type variable. It's all 0s and 1s now.




#%%
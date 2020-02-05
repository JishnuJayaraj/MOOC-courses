# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 19:54:17 2019


week 3: Exploratory data analysis(coursera)

https://labs.cognitiveclass.ai/tools/jupyterlab/lab/tree/labs/DA0101EN/exploratory-data-analysis.ipynb

"""
#%%
import pandas as pd
import numpy as np

#load data and store in dataframe df
path='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)
df.head()

import matplotlib.pyplot as plt
import seaborn as sns

# understand what type of variable you are dealing with. This will help us find the right visualization method for that variable.
# list the data types for each column
print(df.dtypes)

#  we can calculate the correlation between variables of type "int64" or "float64" using the method "corr":
df.corr()

#Find the correlation between the following columns: bore, stroke,compression-ratio , and horsepower.
df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr() 

# Continuous numerical variables are variables that may contain any value within some range. Continuous numerical variables can have the type "int64" or "float64". 
# A great way to visualize these variables is by using scatterplots with fitted lines.

# Let's find the scatterplot of "engine-size" and "price"
# Engine size as potential predictor variable of price
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)

# We can examine the correlation between 'engine-size' and 'price' and see it's approximately 0.87
df[["engine-size", "price"]].corr()

# Let's see if "Peak-rpm" as a predictor variable of "price".
sns.regplot(x="peak-rpm", y="price", data=df)
df[['peak-rpm','price']].corr()
# weak linear relationship

# -- A good way to visualize categorical variables is by using boxplots.

# Let's look at the relationship between "body-style" and "price".
sns.boxplot(x="body-style", y="price", data=df)
#distributions of price between the different body-style categories have a significant overlap, and so body-style would not be a good predictor of price.

sns.boxplot(x="engine-location", y="price", data=df)
#  distribution of price between these two engine-location categories, front and rear, are distinct enough to take engine-location as a potential good predictor of price.








#%%

# --------- Descriptive statistical Analysis

#%%
df.describe()
# The default setting of "describe" skips variables of type object. We can apply the method "describe" on the variables of type 'object' as follows:
df.describe(include=['object'])

# -- Value counts
# Value-counts is a good way of understanding how many units of each characteristic/variable we have
# "value_counts" only works on Pandas series, not Pandas Dataframes.
# we only include one bracket "df['drive-wheels']" not two brackets "df[['drive-wheels']]".

df['drive-wheels'].value_counts()

# convert the series to a Dataframe as follows
df['drive-wheels'].value_counts().to_frame()


#  repeat the above steps but save the results to the dataframe "drive_wheels_counts" and rename the column 'drive-wheels' to 'value_counts'.
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts

#  let's rename the index to 'drive-wheels'
drive_wheels_counts.index.name = 'drive-wheels'
drive_wheels_counts

#We can repeat the above process for the variable 'engine-location'.
# engine-location as variable
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts.head(10)


#%%

# -------------- BASIC GROUPING


#%%

# groups data by different categories. The data is grouped based on one or several variables and analysis is performed on the individual groups.

# let's group by the variable "drive-wheels"
df['drive-wheels'].unique()



#  select the columns 'drive-wheels', 'body-style' and 'price', then assign it to the variable "df_group_one".
df_group_one = df[['drive-wheels','body-style','price']]

#  calculate the average price for each of the different categories of data.
# grouping results
df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()
df_group_one


# group with multiple variables.
# let's group by both 'drive-wheels' and 'body-style'. 

df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
grouped_test1

# easier to visualize when it is made into a pivot table
grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
grouped_pivot

# We can fill these missing cells with the value 0
grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0
grouped_pivot



#use the grouped results, fora heat map visualisation
plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
plt.show()
# The heatmap plots the target variable (price) proportional to colour with respect to the variables 'drive-wheel' and 'body-style' in the vertical and horizontal axis respectively.

#default labels convey no useful information to us. Let's change that:
fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()



#%%



# --------------------------- Correlation and Causation


#%%

# Correlation: a measure of the extent of interdependence between variables.
# Causation: the relationship between cause and effect between two variables.

# The Pearson Correlation measures the linear dependence between two variables X and Y

from scipy import stats

#Let's calculate the Pearson Correlation Coefficient and P-value of 'wheel-base' and 'price'.
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)






#%%

# -------------------------ANOVA: Analysis of Variance

#%%
# statistical method used to test whether there are significant differences between the means of two or more groups.

# F-test score: ANOVA assumes the means of all groups are the same, calculates how much the actual means deviate from the assumption, and reports it as the F-test score. A larger score means there is a larger difference between the means
# P-value: P-value tells how statistically significant is our calculated score value.


# If our price variable is strongly correlated with the variable we are analyzing, expect ANOVA to return a sizeable F-test score and a small p-value


# Let's see if different types 'drive-wheels' impact 'price', we group the data
grouped_test2=df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])
grouped_test2.head(2)

# We can obtain the values of the method group using the method "get_group".
grouped_test2.get_group('4wd')['price']

#use the function 'f_oneway' in the module 'stats' to obtain the F-test score and P-value.
# ANOVA
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val)   

# Separately: fwd and rwd¶
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val )


#%%
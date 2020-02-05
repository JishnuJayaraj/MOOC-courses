# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 15:11:47 2019

@author: test
"""


#%%
# Check the Python Version

import sys

print('Hello, Python!')
print(sys.version)

# System settings about float type
print(sys.float_info)

# Verify that this is an integer
type(2)

# Convert 2 to a float
float(2)

# Note that strings can be represented with single quotes ('1.2') or double quotes ("1.2"), but you can't mix both (e.g., "1.2').
int('1')

# Convert an integer to a string
str(1)

# Division operation expression
25 / 6

# Integer division operation expression
25 // 6

#%%

## ------------ Strings

#%%

# Use quotation marks for defining string

Name = "Michael Jackson"
name = 'Michael Jackson'

# Print the first element in the string
print(Name[0])

# Print the last element in the string
print(Name[-1])

# Find the length of string
len("Michael Jackson")

# Take the slice on variable Name with only index 0 to index 3
Name[0:4]

# Take the slice on variable Name with only index 8 to index 11
Name[8:12]

# Get every second element. The elments on index 1, 3, 5 ...
Name[::2]

# Get every second element in the range from index 0 to index 4
Name[0:5:2]



# Concatenate two strings
Statement = Name + "is the best"
Statement

# Print the string for 3 times
3 * "Michael Jackson"

# New line escape sequence
print(" Michael Jackson \n is the best" )


# Convert all the characters in string to upper case
A = "Thriller is the sixth studio album"
print("before upper:", A)
B = A.upper()
print("After upper:", B)

# Replace the old substring with the new target substring is the segment has been found in the string
A = "Michael Jackson is the best"
B = A.replace('Michael', 'Janet')
print(B)

#The method find finds a sub-string. The argument is the substring you would like to find, and the output is the first index of the sequence. We can find the sub-string jack or el.

# Find the substring in the string. Only the index of the first elment of substring in string will be the output
Name = "Michael Jackson"
Name.find('el')

# If cannot find the substring in the string, output is a negative one
Name.find('Jasdfasdasdf')



#%%





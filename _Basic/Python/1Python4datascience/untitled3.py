# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 00:47:57 2019

@author: test
week 4: read write and basic libraies
"""

# ------------- Reading
#%%
# Download Example file

#!wget -O /resources/data/Example1.txt https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/labs/example1.txt
# Read the Example1.txt

example1 = "/resources/data/Example1.txt"
file1 = open(example1, "r")

# Print the path of file

file1.name

# Print the mode of file, either 'r' or 'w'

file1.mode

# Read the file

FileContent = file1.read()

# Print the file with '\n' as a new line

print(FileContent)

# Type of file content

type(FileContent)

# Close file after finish

file1.close()

# Using the with statement is better practice, it automatically closes the file even if the code encounters an exception. The code will run everything in the indent block then close the file object.


# Open file using with

with open(example1, "r") as file1:
    FileContent = file1.read()
    print(FileContent)
    
# Read first four characters

with open(example1, "r") as file1:
    print(file1.read(4))
    
# Read certain amount of characters

with open(example1, "r") as file1:
    print(file1.read(16))
    print(file1.read(5))
    print(file1.read(9))
    
# Read one line

with open(example1, "r") as file1:
    print("first line: " + file1.readline())
    
# Iterate through the lines

with open(example1,"r") as file1:
        i = 0;
        for line in file1:
            print("Iteration", str(i), ": ", line)
            i = i + 1;
        
        
        
# Read all lines and save as a list
with open(example1, "r") as file1:
    FileasList = file1.readlines()

# Print the first line
FileasList[0]








#%%


# --------------- Writing
https://www.coursera.org/learn/python-for-applied-data-science/lecture/QN5Cl/loading-data-with-pandas
#%%


#%%


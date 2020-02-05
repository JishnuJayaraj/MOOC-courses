# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 22:43:54 2019

@author: test

Week 3
"""
# conditions

#%%
# Use Equality sign to compare the strings

"ACDC" == "Michael Jackson"


# Elif statment example

age = 18

if age > 18:
    print("you can enter" )
elif age == 18:
    print("go see Pink Floyd")
else:
    print("go see Meat Loaf" )
    
print("move on")


# Condition statement example

album_year = 1980

if(album_year > 1979) and (album_year < 1990):  # also  "or"
    print ("Album year was in between 1980 and 1989")
    
print("")
print("Do Stuff..")



#%%

# ------------- Loops

#%%

# Use the range
range(3)

# For loop example
dates = [1982,1980,1973]
N = len(dates)

for i in range(N):
    print(dates[i])  
    
# Loop through the list and iterate on both index and element value
squares=['red', 'yellow', 'green', 'purple', 'blue']

for i, square in enumerate(squares):
    print(i, square)    


#------------ While Loop Example

dates = [1982, 1980, 1973, 2000]

i = 0
year = 0

while(year != 1973):
    year = dates[i]
    i = i + 1
    print(year)

print("It took ", i ,"repetitions to get out of loop.")


#%%


# ------------- functions


#%%
# sorted is a function and returns a new list, it does not change the list L
# First function example: Add 1 to a and store as b

def add(a):
    b = a + 1
    print(a, "if you add one", b)
    return(b)
    
# Get a help on add function
help(add)

# Define a function for multiple two numbers
def Mult(a, b):
    c = a * b
    return(c)



# -------------------- pre-defined functions in Python
    
# Build-in function print()

album_ratings = [10.0, 8.5, 9.5, 7.0, 7.0, 9.5, 9.0, 9.5] 
print(album_ratings)

# Use sum() to add every element in a list or tuple together
sum(album_ratings)

# Show the length of the list or tuple
len(album_ratings)





# Example for setting param with default value

def isGoodRating(rating=4): 
    if(rating < 7):
        print("this album sucks it's rating is",rating)
        
    else:
        print("this album is good its rating is",rating)
        

#%%
        
# ----------- Class      
        
#%%
# Import the library

import matplotlib.pyplot as plt

# Create a class Circle

class Circle(object):
    
    # Constructor
    def __init__(self, radius=3, color='blue'):
        self.radius = radius
        self.color = color 
    
    # Method
    def add_radius(self, r):
        self.radius = self.radius + r
        return(self.radius)
    
    # Method
    def drawCircle(self):
        plt.gca().add_patch(plt.Circle((0, 0), radius=self.radius, fc=self.color))
        plt.axis('scaled')
        plt.show() 
        
        
# Create an object RedCircle
RedCircle = Circle(10, 'red')

# Find out the methods can be used on the object RedCircle
dir(RedCircle)

# Print the object attribute radius
RedCircle.radius

# Set the object attribute radius
RedCircle.radius = 1
        
# Call the method drawCircle
RedCircle.drawCircle()

# Use method to change the object attribute radius
print('Radius of object:',RedCircle.radius)
RedCircle.add_radius(2)
print('Radius of object of after applying the method add_radius(2):',RedCircle.radius)
RedCircle.add_radius(5)
print('Radius of object of after applying the method add_radius(5):',RedCircle.radius)

# Create a blue circle with a given radius
BlueCircle = Circle(radius=100)

# Call the method drawCircle
BlueCircle.drawCircle()

# Create a new Rectangle class for creating a rectangle object

class Rectangle(object):
    
    # Constructor
    def __init__(self, width=2, height=3, color='r'):
        self.height = height 
        self.width = width
        self.color = color
    
    # Method
    def drawRectangle(self):
        plt.gca().add_patch(plt.Rectangle((0, 0), self.width, self.height ,fc=self.color))
        plt.axis('scaled')
        plt.show()
        
        
        
        
# Create a new object rectangle

SkinnyBlueRectangle = Rectangle(2, 10, 'blue')

# Print the object attribute height

SkinnyBlueRectangle.height 

# Use the drawRectangle method to draw the shape

SkinnyBlueRectangle.drawRectangle()

# Create a new object rectangle

FatYellowRectangle = Rectangle(20, 5, 'yellow')

# Use the drawRectangle method to draw the shape

FatYellowRectangle.drawRectangle()






#%%
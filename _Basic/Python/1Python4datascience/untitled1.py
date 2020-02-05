# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 19:02:34 2019

@author: test
"""
#%%
# Create your first tuple
tuple1 = ("disco",10,1.2 )
tuple1
# Print the type of the tuple you created
print(type(tuple1))

# Print the variable on each index
print(tuple1[0])
print(tuple1[1])
print(tuple1[2])

# Print the type of value on each index
print(type(tuple1[0]))
print(type(tuple1[1]))
print(type(tuple1[2]))

# Use negative index to get the value of the last element
tuple1[-1]
tuple1[-2]
tuple1[-3]

# Concatenate two tuples
tuple2 = tuple1 + ("hard rock", 10)
tuple2

# Slice from index 0 to index 2
tuple2[0:3]

# Get the length of tuple
len(tuple2)

# ---- Sorting

# A sample tuple
Ratings = (0, 9, 6, 5, 10, 8, 9, 6, 2)

# Sort the tuple
RatingsSorted = sorted(Ratings)
RatingsSorted

# ---- Nested Tuple
# Create a nest tuple
NestedT =(1, 2, ("pop", "rock") ,(3,4),("disco",(1,2)))

# Print element on each index
print("Element 0 of Tuple: ", NestedT[0])
print("Element 1 of Tuple: ", NestedT[1])
print("Element 2 of Tuple: ", NestedT[2])
print("Element 3 of Tuple: ", NestedT[3])
print("Element 4 of Tuple: ", NestedT[4])

# Print element on each index, including nest indexes
print("Element 2, 0 of Tuple: ",   NestedT[2][0])
print("Element 2, 1 of Tuple: ",   NestedT[2][1])
print("Element 3, 0 of Tuple: ",   NestedT[3][0])
print("Element 3, 1 of Tuple: ",   NestedT[3][1])
print("Element 4, 0 of Tuple: ",   NestedT[4][0])
print("Element 4, 1 of Tuple: ",   NestedT[4][1])

# Print the first element in the second nested tuples
NestedT[2][1][0]






#%%




#------------------- Lists (mutable)

#%%

# Create a list
L = ["Michael Jackson", 10.1, 1982]


# Print the elements on each index
print('the same element using negative and positive indexing:\n Postive:',L[0],
'\n Negative:' , L[-3]  )
print('the same element using negative and positive indexing:\n Postive:',L[1],
'\n Negative:' , L[-2]  )
print('the same element using negative and positive indexing:\n Postive:',L[2],
'\n Negative:' , L[-1]  )



# Use extend to add elements to list
L.extend(['pop', 10])

# Use append to add elements to list(we add one element to the list)
L.append(['pop', 10])

# List slicing
print(L[3:5])


# Change the element based on the index
A = ["disco", 10, 1.2]
print('Before change:', A)
A[0] = 'hard rock'
print('After change:', A)

# Delete the element based on the index
print('Before change:', A)
del(A[0])
print('After change:', A)

#  convert a string to a list using split. 

# Split the string, default is by space
'hard rock'.split()
# Split the string by comma
'A,B,C,D'.split(',')

# Copy (copy by reference) the list A
A = ["hard rock", 10, 1.2]
B = A
print('A:', A)
print('B:', B)
# Examine the copy by reference
print('B[0]:', B[0])
A[0] = "banana"
print('B[0]:', B[0])

# Clone (clone by value) the list A
B = A[:]

print('B[0]:', B[0])
A[0] = "hard rock"
print('B[0]:', B[0])





#%%


# ----------------- Dictionaries

#%%

# Create the dictionary

Dict = {"key1": 1, "key2": "2", "key3": [3, 3, 3], "key4": (4, 4, 4), ('key5'): 5, (0, 1): 6}

# Access to the value by the key
Dict["key1"]

# Access to the value by the key
Dict[(0, 1)]

# Create a sample dictionary
release_year_dict = {"Thriller": "1982", "Back in Black": "1980", \
                    "The Dark Side of the Moon": "1973", "The Bodyguard": "1992", \
                    "Bat Out of Hell": "1977", "Their Greatest Hits (1971-1975)": "1976", \
                    "Saturday Night Fever": "1977", "Rumours": "1977"}

# Get value by keys
release_year_dict['Thriller'] 

# Get all the keys in dictionary
release_year_dict.keys() 


# Get all the values in dictionary
release_year_dict.values() 

# Append value with key into dictionary
release_year_dict['Graduation'] = '2007'

# Delete entries by key
del(release_year_dict['Thriller'])
del(release_year_dict['Graduation'])

# Verify the key is in the dictionary
'The Bodyguard' in release_year_dict



#%%

# --------------- Sets

#%%
# A set is a unique collection of objects in Python

# Create a set
set1 = {"pop", "rock", "soul", "hard rock", "rock", "R&B", "rock", "disco"}

# Convert list to set
album_list = [ "Michael Jackson", "Thriller", 1982, "00:42:19", \
              "Pop, Rock, R&B", 46.0, 65, "30-Nov-82", None, 10.0]
album_set = set(album_list)

# Convert list to set
music_genres = set(["pop", "pop", "rock", "folk rock", "hard rock", "soul", \
                    "progressive rock", "soft rock", "R&B", "disco"])


# Sample set
A = set(["Thriller", "Back in Black", "AC/DC"])

# Add element to set
A.add("NSYNC")

# Remove the element from set
A.remove("NSYNC")

# Verify if the element is in the set
"AC/DC" in A

# ------ Sets Logic Operations

# Sample Sets

album_set1 = set(["Thriller", 'AC/DC', 'Back in Black'])
album_set2 = set([ "AC/DC", "Back in Black", "The Dark Side of the Moon"])

# Find the intersections

intersection = album_set1 & album_set2


# Find the difference in set1 but not set2

album_set1.difference(album_set2)


# Use intersection method to find the intersection of album_list1 and album_list2

album_set1.intersection(album_set2)

# Find the union of two sets

album_set1.union(album_set2)

# Check if superset

set(album_set1).issuperset(album_set2)  

# Check if subset

set(album_set2).issubset(album_set1) 




#%%







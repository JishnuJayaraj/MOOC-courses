# LISTS 

 colors = ['red', 'blue', 'green']
  print colors[0]    ## red
  print colors[2]    ## green
  print len(colors)  ## 3
  
   b = colors   ## Does not copy the list
   # Assignment with an = on lists does not make a copy. Instead, assignment makes the two variables point to the one list in memory.
   
   colors2 = ['black','white']
   sumcolors = colors + colors2
   print sumcolors
   
   
   
   # for var in list:       : is an easy way to look at each element in a list (or other collection).
  squares = [1, 4, 9, 16]
  sum = 0
  for num in squares:
    sum += num
  print sum  ## 30
  
  # value in collection:    :  easy way to test if an element appears in a list (or other collection) 
  list = ['larry', 'curly', 'moe']
  if 'curly' in list:
    print 'yay'
	
	
  # range(n)                : The range(n) function yields the numbers 0, 1, ... n-1
  # print the numbers from 0 through 99
  for i in range(100):
    print i
	
	
  # Access every 3rd element in a list
  i = 0
  while i < len(a):
    print a[i]
    i = i + 3
	
	
	
	
	
	# METHODSSSSS
	
	"""
	list.append(elem) -- adds a single element to the end of the list
	list.insert(index, elem) -- inserts the element at the given index, shifting elements to the right.
	list.extend(list2) adds the elements in list2 to the end of the list. Using + or += on a list is similar to using extend().
	list.index(elem) -- searches for the given element from the start of the list and returns its index
	list.remove(elem) -- searches for the first instance of the given element and removes it
	list.pop(index) -- removes and returns the element at the given index
	"""
	
  list = ['larry', 'curly', 'moe']
  list.append('shemp')         ## append elem at end
  list.insert(0, 'xxx')        ## insert elem at index 0
  list.extend(['yyy', 'zzz'])  ## add list of elems at end
  print list  ## ['xxx', 'larry', 'curly', 'moe', 'shemp', 'yyy', 'zzz']
  print list.index('curly')    ## 2

  list.remove('curly')         ## search and remove that element
  list.pop(1)                  ## removes and returns 'larry'
  print list  ## ['xxx', 'moe', 'shemp', 'yyy', 'zzz']
  
  
  list = [1, 2, 3]
  print list.append(4)   ## NO, does not work, append() returns None
  ## Correct pattern:
  list.append(4)
  print list  ## [1, 2, 3, 4]
  
  # build up
  list = []          ## Start as the empty list
  list.append('a')   ## Use append() to add elements
  list.append('b')
  
  # slices
  list = ['a', 'b', 'c', 'd']
  print list[1:-1]   ## ['b', 'c']
  list[0:2] = 'z'    ## replace ['a', 'b'] with ['z']
  print list         ## ['z', 'c', 'd']
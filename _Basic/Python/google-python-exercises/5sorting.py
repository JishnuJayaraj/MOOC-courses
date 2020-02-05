  a = [5, 1, 4, 3]
  print sorted(a)                    ## [1, 3, 4, 5]
  print a                            ## [5, 1, 4, 3]
  
  strs = ['aa', 'BB', 'zz', 'CC']
  print sorted(strs)                 ## ['BB', 'CC', 'aa', 'zz'] (case sensitive)
  print sorted(strs, reverse=True)   ## ['zz', 'aa', 'CC', 'BB']
  
  
 # custom
  strs = ['ccc', 'aaaa', 'd', 'bb']
  print sorted(strs, key=len)        ## ['d', 'bb', 'ccc', 'aaaa']
  
  # "key" argument specifying str.lower function to use for sorting
  print sorted(strs, key=str.lower)  ## ['aa', 'BB', 'CC', 'zz']
  
  
  # Say we have a list of strings we want to sort by the last letter of the string.
  strs = ['xc', 'zb', 'yd' ,'wa']

  # Write a little function that takes a string, and returns its last letter.
  # This will be the key function (takes in 1 value, returns 1 value).
  def MyFn(s):
    return s[-1]

  # Now pass key=MyFn to sorted() to sort by the last letter:
  print sorted(strs, key=MyFn)       ## ['wa', 'zb', 'xc', 'yd']
  
  
 # TUPLESSSS
 
 """
 A tuple is a fixed size grouping of elements, such as an (x, y) co-ordinate. Tuples are like lists, except they are immutable and do not change size.
 play a sort of "struct" role in Python -way to pass around a little logical, fixed size bundle of values
 """
 
  tuple = (1, 2, 'hi')
  print len(tuple) 		 		 ## 3
  print tuple[2]   				 ## hi
  tuple[2] = 'bye' 				 ## NO, tuples cannot be changed
  tuple = (1, 2, 'bye') 		 ## this works
  
  tuple = ('hi',)  				 ## size-1 tuple
  
  # list comprehensions
  nums = [1, 2, 3, 4]

  squares = [ n * n for n in nums ]   ## [1, 4, 9, 16]
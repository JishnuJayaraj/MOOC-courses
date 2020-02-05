  #Python has a built-in string class named "str" with many handy features
  # Python strings are "immutable" which means they cannot be changed after they are created
  # The [ ] syntax and the len() function actually work on any sequence type -- strings, lists, etc.
  
  s = 'hi'
  print s[1]          ## i
  print len(s)        ## 2
  print s + ' there'  ## hi there
  
  pi = 3.14
  # text = 'The value of pi is ' + pi      ## NO, does not work
  text = 'The value of pi is '  + str(pi)  ## yes
  
  # If you want integer division, it is most correct to use 2 slashes -- e.g. 6 // 5 is 1
  
  raw = r'this\t\n and that'

  # this\t\n and that
  print raw

  multi = """It was the best of times.
  It was the worst of times."""

  # It was the best of times.
  #   It was the worst of times.
  print multi
  
  
  # METHODS
  # http://docs.python.org/library/stdtypes.html#string-methods
  
  
  # The "slice" syntax is a handy way to refer to sub-parts of sequences
  # The slice s[start:end] is the elements beginning at start and extending up to but not including end.
  
  # H  e  l  l  o                        s[1:4] is 'ell' 
  # 0  1  2  3  4                        s[1:] is 'ello'
  #-5 -4 -3 -2 -1                        s[:] is 'Hello'   pythonic way to copy a sequence
  
  
    # % operator
  text = "%d little pigs come out, or I'll %s, and I'll %s, and I'll blow your %s down." % (3, 'huff', 'puff', 'house')
  
  # suppose you want to break it into separate lines. enclose the whole expression in an outer set of parenthesis -- then the expression is allowed to span multiple lines.
  
    # Add parentheses to make the long line work:
  text = (
    "%d little pigs come out, or I'll %s, and I'll %s, and I'll blow your %s down."
    % (3, 'huff', 'puff', 'house'))
	
    # Split the line into chunks, which are concatenated automatically by Python
  text = (
    "%d little pigs come out, "
    "or I'll %s, and I'll %s, "
    "and I'll blow your %s down."
    % (3, 'huff', 'puff', 'house'))
	
	
	
	
  if speed >= 80:
    print 'License and registration please'
    if mood == 'terrible' or speed >= 100:
      print 'You have the right to remain silent.'
    elif mood == 'bad' or speed >= 90:
      print "I'm going to have to write you a ticket."
      write_ticket()
    else:
      print "Let's try to keep it under 80 ok?"
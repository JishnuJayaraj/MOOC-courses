# Regular expressions are a powerful language for matching text patterns
  match = re.search(pat, str)  # regular exp search
  
  
    str = 'an example word:cat!!'
	match = re.search(r'word:\w\w\w', str)
	# If-statement after search() tests if it succeeded
	if match:
	  print 'found', match.group()             ## 'found word:cat'
	else:
	  print 'did not find'
	  
	  # The 'r' at the start of the pattern string designates a python "raw" string which passes through backslashes without change which is very handy for regular expressions
	  
	  
	  
	  
	  
  ## Search for pattern 'iii' in string 'piiig'.
  ## All of the pattern must match, but it may appear anywhere.
  ## On success, match.group() is matched text.
  match = re.search(r'iii', 'piiig') # found, match.group() == "iii"
  match = re.search(r'igs', 'piiig') # not found, match == None

  ## . = any char but \n
  match = re.search(r'..g', 'piiig') # found, match.group() == "iig"

  ## \d = digit char, \w = word char
  match = re.search(r'\d\d\d', 'p123g') # found, match.group() == "123"
  match = re.search(r'\w\w\w', '@@abcd!!') # found, match.group() == "abc"
  
  
 
# -----------------------------------------------------------------
 
  ## i+ = one or more i's, as many as possible.
  match = re.search(r'pi+', 'piiig') # found, match.group() == "piii"

  ## Finds the first/leftmost solution, and within it drives the +
  ## as far as possible (aka 'leftmost and largest').
  ## In this example, note that it does not get to the second set of i's.
  match = re.search(r'i+', 'piigiiii') # found, match.group() == "ii"

  ## \s* = zero or more whitespace chars
  ## Here look for 3 digits, possibly separated by whitespace.
  match = re.search(r'\d\s*\d\s*\d', 'xx1 2   3xx') # found, match.group() == "1 2   3"
  match = re.search(r'\d\s*\d\s*\d', 'xx12  3xx') # found, match.group() == "12  3"
  match = re.search(r'\d\s*\d\s*\d', 'xx123xx') # found, match.group() == "123"

  ## ^ = matches the start of string, so this fails:
  match = re.search(r'^b\w+', 'foobar') # not found, match == None
  ## but without the ^ it succeeds:
  match = re.search(r'b\w+', 'foobar') # found, match.group() == "bar"
  
  # email example
  
  str = 'purple alice-b@google.com monkey dishwasher'
  match = re.search(r'\w+@\w+', str)
  if match:
    print match.group()  						## 'b@google'
	
	
  match = re.search(r'[\w.-]+@[\w.-]+', str)
  if match:
    print match.group()  						## 'alice-b@google.com'
	
	
# Group extraction

  str = 'purple alice-b@google.com monkey dishwasher'
  match = re.search(r'([\w.-]+)@([\w.-]+)', str)
  if match:
    print match.group()  						 ## 'alice-b@google.com' (the whole match)
    print match.group(1) 						 ## 'alice-b' (the username, group 1)
    print match.group(2) 						 ## 'google.com' (the host, group 2)
	
	
	
# find all
  ## Suppose we have a text with many email addresses
  str = 'purple alice@google.com, blah monkey bob@abc.com blah dishwasher'

  ## Here re.findall() returns a list of all the found email strings
  emails = re.findall(r'[\w\.-]+@[\w\.-]+', str) ## ['alice@google.com', 'bob@abc.com']
  for email in emails:
    # do something with each found email string
    print email
	
	
# Find all with files
  # Open file
  f = open('test.txt', 'r')
  # Feed the file text into findall(); it returns a list of all the found strings
  strings = re.findall(r'some pattern', f.read())
  
  
  str = 'purple alice@google.com, blah monkey bob@abc.com blah dishwasher'
  tuples = re.findall(r'([\w\.-]+)@([\w\.-]+)', str)
  print tuples  ## [('alice', 'google.com'), ('bob', 'abc.com')]
  for tuple in tuples:
    print tuple[0]  ## username
    print tuple[1]  ## host
# Defines a "repeat" function that takes 2 arguments.
def repeat(s, exclaim):
    """
    Returns the string 's' repeated 3 times.
    If exclaim is true, add exclamation marks.
    """

    result = s + s + s # can also use "s * 3" which is faster (Why?)
    if exclaim:
        result = result + '!!!'
    return result
	
def main():
    print repeat('Yay', False)      ## YayYayYay
    print repeat('Woo Hoo', True)   ## Woo HooWoo HooWoo Hoo!!!
	

  
  
 # Modules and namespaces
"""
Suppose you've got a module "binky.py" which contains a "def foo()". The fully qualified name of that foo function is "binky.foo". In this way, various Python modules can name their functions and variables whatever they want, and the variable names won't conflict — module1.foo is different from module2.foo. In the Python vocabulary, we'd say that binky, module1, and module2 each have their own "namespaces," which as you can guess are variable name-to-object bindings.
""" 

  import sys

  # Now can refer to sys.xxx facilities
  sys.exit(0)


#  Standard Library modules and packages at http://docs.python.org/library.


# help(sys) — help string for the sys module (must do an import sys first)
# dir(sys) — dir() is like help() but just gives a quick list of its defined symbols, or "attributes"
# help(sys.exit)

if __name__ == '__main__':
  main()
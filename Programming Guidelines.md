# Programming Guidelines

## Module Order

1. **Module Docstring**<br>
Start with a module-level docstring that explains the purpose and content of the module. This is good practice for readability and maintainability.

2. **Imports**

	List all your import statements next, grouped in the following order:
   - Standard library imports
     - Related third party imports
     - Local application/library specific imports
     Make sure to separate these groups by a blank line.

3. **Constants**<br>
Define any constants that your class or classes within the module might use.
Capital letters for Magic Numbers


## Classes
For each class, you can follow this order:


### Class Order

1.  **Class Docstring**<br>
	Start with a docstring immediately after the class definition to describe the purpose and behavior of the class.

2. **Class Variables**<br>
	- Protected Class Variables (_var):<br>Follow with variables that should not be accessed directly outside the class.
	- Public Class Variables:<br>Start with variables that are shared across all instances of a class.
	
3. **_ _ init _ _**<br>
	- asserts and super() call first
    - Property assignments:
      - Protected before public
      - Type order 
        1. Auto-attribute assignment
        2. Init of vars with value None
        3. Init of vars with null-ish values like zero or ''.
        4. Other vars
	- Method calls last

4. **Magic Methods**<br>
	After _ _ init _ _ list other dunder methods like _ _ str _ _, _ _ repr _ _, _ _ len _ _.

6. **Metods**
    - Abstract Methods (@abstractmethod): Virtual methods to be overridden by children
	- Static Methods (@staticmethod): Methods that don't access class or instance-specific data.
	- Class Methods (@classmethod): Methods that access class variables but not instance variables.
	- Protected Methods (_method and _ _method): Methods for internal use within the class. Start with _method for protected methods and __method for private methods.
	- Public Methods: Define these next. They are the methods intended to be accessed from outside the class.
   
5. **Properties**

   @property and Setters

**Functions**

Do Type checking of all variables and returns. Thank me later.

	

**Final Notes**
	
Use PEP-8 as far as feasible.	

Build modular.

Prefer to return object (Holders child or namedtuple) rather than multiple return values 

Keep things consistent. Keep similar code similar.